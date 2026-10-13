from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class SWFontModel(ModelMixin, ConfigMixin):
    """SW-Font model integrating width-aware style encoding and skeleton-guided denoising."""

    @register_to_config
    def __init__(
            self,
            unet,
            style_encoder,
            content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

    def forward(
            self,
            x_t,
            timesteps,
            style_images=None,
            content_images=None,
            style_width_images=None,
            content_skeleton_images=None,
            content_encoder_downsample_size=3,
            cond=None,
            **kwargs,
    ):
        if cond is not None:
            style_images = cond[0]
            content_images = cond[1]
            if len(cond) >= 4:
                style_width_images = cond[2]
                content_skeleton_images = cond[3]

        if style_images is None or content_images is None:
            raise ValueError("SWFontModel requires either `cond` or both `style_images` and `content_images`.")

        # ==========================================================
        # ==========================================================
        style_img_feature, _, style_residual_features = self.style_encoder(
            style_rgb=style_images,
            style_width=style_width_images
        )

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height * width, channel)

        # ==========================================================
        # ==========================================================

        final_content_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(final_content_feature)

        final_style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(final_style_content_feature)

        input_hidden_states = [
            style_img_feature,  # index 0: Tensor
            content_residual_features,
            style_hidden_states,  # index 2: Tensor
            style_content_res_features
        ]

        # ==========================================================
        # ==========================================================
        out = self.unet(
            sample=x_t,
            timestep=timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,

            current_content_feature=final_content_feature,

            skeleton_map=content_skeleton_images
        )

        noise_pred = out[0] if isinstance(out, tuple) else out.sample

        return noise_pred
