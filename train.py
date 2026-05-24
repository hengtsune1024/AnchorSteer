import logging
import math
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn.functional as F

from models import Controller, load_pipeline

from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers import DDPMScheduler

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from config import parse_args
from utils_data import get_dataloader

logger = get_logger(__name__)

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    yaml = YAML()
    yaml.dump(vars(args), open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    ############# Model #############
    logger.info("Loading Stable audio Pipeline")
    # Load to CPU first; only the submodules needed for preprocessing are moved to the
    # accelerator device. The transformer stays on CPU until accelerator.prepare() below,
    # which handles device placement and optional DDP wrapping.
    pipe = load_pipeline("interpret", dtype=torch.float16, device="cpu")

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.projection_model.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    pipe.vae.to(accelerator.device)
    pipe.text_encoder.to(accelerator.device)
    pipe.projection_model.to(accelerator.device)

    # SAO uses v-prediction; the DDPM scheduler config is re-used with that prediction type
    # so targets are velocity vectors rather than raw noise.
    noise_scheduler = DDPMScheduler.from_config(
        pipe.scheduler.config,
        prediction_type="v_prediction"
    )

    ############# Controller #############
    num_hidden_vec = (1024 if args.train_full_len else 107) + (1 if args.inject_first_state else 0)
    controller = Controller(
        edit_layers=list(range(args.edit_start, args.edit_end+1)),
        use_time=args.use_time_emb,
        total_layers=len(pipe.transformer.transformer_blocks),
        control_type=args.control_type,
        inject_first_state=args.inject_first_state,
        hidden_dim=(num_hidden_vec, 1536)
    )
    pipe.set_controller(controller)
    print(f"Controller total Parameters: {sum(p.numel() for p in controller.parameters())}")
    print(f"Controller dimension: {(num_hidden_vec, 1536)}")
    print(controller)

    ############# Optimizer #############
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    ############# Dataloader #############
    train_dataloader = get_dataloader(
        args.train_data_dir,
        pipe=pipe,
        mini_batch_size=args.mini_batch_size,
        encode_batch_size=args.encode_batch_size,
        num_workers=4,
        shuffle=True,
    )
    # VAE / text encoder / projection model are only needed to pre-encode the dataset.
    # After get_dataloader returns (all latents are cached in RAM), freeing them recovers
    # several GB of VRAM before the training loop begins.
    del pipe.vae
    del pipe.text_encoder
    del pipe.projection_model

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # The entire transformer (not just the controller) is wrapped by accelerator so that
    # device placement and potential DDP sharding are handled correctly, even though only
    # controller parameters receive gradients.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipe.transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Recompute after accelerator.prepare because distributed sharding may change the
    # effective dataset length seen by each process.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="anchorsteer",
            config={k: v for k, v in vars(args).items() if k != 'config'},
        )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    loss_history = []
    epoch_loss_history = []
    best_loss = float('inf')
    global_step = 0

    print("Maximum timestep: ", pipe.scheduler.config.num_train_timesteps)

    # Rotary positional embedding depends only on sequence length (fixed for a given
    # train_full_len setting), so it is computed once here and reused every step.
    latents, _, audio_duration_embeds = next(iter(train_dataloader))
    seq_len = latents.shape[2] + audio_duration_embeds.shape[1]
    rotary_embed_dim = pipe.transformer.config.attention_head_dim // 2
    rotary_embedding = get_1d_rotary_pos_embed(
        rotary_embed_dim,
        seq_len,
        use_real=True,
        repeat_interleave_real=False
    )
    rotary_embedding = tuple(
        t.to(device=pipe.device) for t in rotary_embedding
    )

    for epoch in range(args.num_train_epochs):
        transformer.train()
        train_loss = []
        epoch_start_global_step = global_step
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):

                latents, text_audio_duration_embeds, audio_duration_embeds = batch
                latents = latents.to(accelerator.device)
                text_audio_duration_embeds = text_audio_duration_embeds.to(accelerator.device)
                audio_duration_embeds = audio_duration_embeds.to(accelerator.device)

                bsz = latents.shape[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                latent_model_input = noisy_latents

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=text_audio_duration_embeds,
                    global_hidden_states=audio_duration_embeds,
                    rotary_embedding=rotary_embedding,
                    return_dict=False,
                    steer=1,
                    do_classifier_free_guidance=False,
                )[0]

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss.append(loss.detach().item())
                accelerator.log({"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                loss_history.append(loss.detach().item())

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if not args.skip_evaluation and (global_step) % args.log_every_steps == 0:
                    unwrapped_transformer = accelerator.unwrap_model(transformer)
                    torch.save(unwrapped_transformer.controller.state_dict(), args.output_dir + '/adaptor.pth')
                    plt.figure()
                    plt.plot(loss_history, label='step_loss')
                    if epoch_loss_history:
                        xs, ys = zip(*epoch_loss_history)
                        plt.plot(xs, ys, 'o-', color='red', label='epoch_avg')
                    plt.legend()
                    plt.savefig(args.output_dir + '/loss_history.png')
                    plt.close()

                if global_step >= args.max_train_steps:
                    break

        if epoch % args.log_every_epochs == 0:
            torch.save(controller.state_dict(), args.output_dir + '/adaptor.pth')

        avg_train_loss = sum(train_loss) / len(train_loss)
        # Place the epoch-average point at the midpoint of its step range so the marker
        # sits over the corresponding region of the step-loss curve in the plot.
        epoch_mid = epoch_start_global_step + (global_step - epoch_start_global_step) / 2.0
        epoch_loss_history.append((epoch_mid, avg_train_loss))
        if best_loss > avg_train_loss:
            best_loss = avg_train_loss
            print(f"Best loss {best_loss:.3f} at epoch {epoch}, saved to best.pth")
            torch.save(controller.state_dict(), args.output_dir + '/best.pth')

    torch.save(controller.state_dict(), args.output_dir + '/adaptor.pth')
    plt.figure()
    plt.plot(loss_history, label='step_loss')
    if epoch_loss_history:
        xs, ys = zip(*epoch_loss_history)
        plt.plot(xs, ys, 'o-', color='red', label='epoch_avg')
    plt.legend()
    plt.savefig(args.output_dir + '/loss_history.png')
    plt.close()

    accelerator.end_training()

if __name__ == "__main__":
    main()
