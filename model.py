import os
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.nn.functional import normalize
from torchvision import models

from einops import repeat
from einops.layers.torch import Rearrange
from einops.einops import reduce


device = "cuda" if torch.cuda.is_available() else "cpu"


# model
# In this cell transformer architechture implemented
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.transformer = ViT(960, 9, 16384, 128, 1, 1, 128)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x2 = repeat(x2, "b c h w -> b c (h 2) (w 2)")
        x3 = repeat(x3, "b c h w -> b c (h 4) (w 4)")
        x4 = repeat(x4, "b c h w -> b c (h 8) (w 8)")
        out1 = torch.cat([x1, x2], dim=1)
        out2 = torch.cat([x3, out1], dim=1)
        out3 = torch.cat([x4, out2], dim=1)
        out = self.transformer(out3)
        map1 = reduce(out[:, :512, :, :], "b c (h 2) (w 2) -> b c h w", "mean")
        map2 = reduce(out[:, 512:768, :, :], "b c (h 2) (w 2) -> b c h w", "mean")
        map3 = reduce(out[:, 768:896, :, :], "b c (h 2) (w 2) -> b c h w", "mean")
        map4 = out[:, 896:960, :, :]
        #                                   #new task today
        print(x5.size(), " : ", map1.size())  #
        x = self.up1(x5, map1)

        print(x.size(), " : ", map2.size())  #
        x = self.up2(x, map2)

        print(x.size(), " : ", map3.size())  #
        x = self.up3(x, map3)

        print(x.size(), " : ", map4.size())  #
        x = self.up4(x, map4)

        logits = self.outc(x)
        logits = nn.Sigmoid()(logits)
        return logits


# model part2
# 2
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, head, heads=2, dim_head=128, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.head = head
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.linear1 = nn.Sequential(nn.Linear(16384, 128), nn.Sigmoid())
        self.linear2 = nn.Sequential(nn.Linear(16384, 128), nn.Sigmoid())
        self.linear3 = nn.Sequential(nn.Linear(16384, 128), nn.Sigmoid())
        self.mlp_head = nn.Sequential(nn.Linear(128, 16384), nn.Sigmoid())

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = x.chunk(3, dim=1)
        q, k, v = qkv
        q = q.reshape(-1, self.head, 16384)
        k = k.reshape(-1, self.head, 16384)
        v = v.reshape(-1, self.head, 16384)
        q = self.linear1(q)
        k = self.linear2(k)
        v = self.linear3(v)
        q = q.reshape(-1, 128, self.head)
        k = k.reshape(-1, self.head, 128)
        v = v.reshape(-1, self.head, 128)
        dots = torch.bmm(q, k) * self.scale
        attn = self.attend(dots)
        out = torch.bmm(v, attn)
        out = self.mlp_head(out)
        out = out.reshape(-1, self.head, 128, 128)

        # out = rearrange(out, 'b n h d -> b n (h d)')
        return out


class Transformer(nn.Module):
    def __init__(self, head, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.conv_output = nn.Sequential(
            nn.Conv2d(3 * head, head, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                head,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )
        # this gives me series multi-head attention

    def forward(self, x):
        # we passed through all this series attention and feedforward neural network to get ultimate output
        for attn, ff in self.layers:
            y = self.conv_output(x)
            x = attn(x) + y
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        input_dim,
        head,
        hidden_dim,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head=128,
        dropout=0.0,
    ):
        super().__init__()

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(960, 3 * head, 3, padding=1), nn.ReLU(inplace=True)
        )
        # this layer convert the image into patches and pass them through a single Linear layer whoes vector size is patch_height*patch_width.
        self.transformer = Transformer(
            head, dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(head, input_dim, 3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
        )
        self.mlp_head = nn.Sequential(nn.Linear(128, 16384), nn.Sigmoid())

    def forward(self, img):
        x = self.to_patch_embedding(
            img
        )  # this give me the desired matrix that come from combination of all patches of transformer
        x = self.transformer(x)
        # x = rearrange(x, 'b c h -> b h c')
        # x = self.mlp_head(x)
        # plt.imshow(x[0][0].detach().cpu().numpy())
        # plt.show()
        out = self.conv_out(x)
        return out


# model next
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


#
from sklearn.metrics import jaccard_score as jsc
from scipy.spatial import distance


def iou(Im1, Im2):
    flat1 = Im1.flatten()
    flat2 = Im2.flatten()
    return jsc(flat1, flat2)


def dice_coeff(Im1, Im2):
    flat1 = Im1.flatten()
    flat2 = Im2.flatten()
    ans = distance.dice(flat1, flat2)
    res = 1 - ans
    return res


# disc
cuda = True if torch.cuda.is_available() else False
criterion_GAN_disc = nn.BCELoss()
criterion_pixelwise_disc = torch.nn.L1Loss()
# criterion_pixelwise = nn.BCELoss()
lambda_pixel = 100
# Calculate output of image discriminator (PatchGAN)
patch = (1, 128 // 2**4, 128 // 2**4)
generator_disc = UNet(1, 1, bilinear=True)
discriminator_disc = Discriminator()

if cuda:
    generator_disc = generator_disc.cuda()
    discriminator_disc = discriminator_disc.cuda()
    criterion_GAN_disc.cuda()
    criterion_pixelwise_disc.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator_disc.parameters(), lr=0.0001)
optimizer_D = torch.optim.Adam(discriminator_disc.parameters())
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
visualize = True


# cup
cuda = True if torch.cuda.is_available() else False
criterion_GAN_cup = nn.BCELoss()
criterion_pixelwise_cup = torch.nn.L1Loss()
# criterion_pixelwise = nn.BCELoss()
lambda_pixel = 100
# Calculate output of image discriminator (PatchGAN)
patch = (1, 128 // 2**4, 128 // 2**4)
generator_cup = UNet(1, 1, bilinear=True)
discriminator_cup = Discriminator()

if cuda:
    generator_cup = generator_cup.cuda()
    discriminator_cup = discriminator_cup.cuda()
    criterion_GAN_cup.cuda()
    criterion_pixelwise_cup.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator_cup.parameters(), lr=0.0001)
optimizer_D = torch.optim.Adam(discriminator_cup.parameters())
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
visualize = True


# classification model
#
classifier3 = models.squeezenet1_1(pretrained=True)
classifier3.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2))
classifier3.classifier = nn.Sequential(
    nn.Dropout(p=0.5, inplace=False),
    nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)),
    nn.Flatten(),
    nn.Linear(49, 1),
    nn.Sigmoid(),
)


criterion_classifier3 = nn.BCELoss()

if cuda:
    classifier3 = classifier3.cuda()
    criterion_classifier3.cuda()

optimizer_C3 = torch.optim.Adam(classifier3.parameters(), lr=0.0001)


# loading checkpoints
checkpoint1 = torch.load(
    r"data\MODEL_WEIGHTS\drishtigs\generator_disc_noaug_new.pth",
    map_location=torch.device(device),
)
generator_disc.load_state_dict(checkpoint1["generator_disc_state_dict"])

checkpoint2 = torch.load(
    r"data\MODEL_WEIGHTS\drishtigs\generator_cup_noaug_new.pth",
    map_location=torch.device(device),
)
generator_cup.load_state_dict(checkpoint2["generator_cup_state_dict"])

checkpoint3 = torch.load(
    r"data\MODEL_WEIGHTS\classifier_noaug_new.pth", map_location=torch.device(device)
)
classifier3.load_state_dict(checkpoint3["classifier3"])


# prediction

transform = transforms.Resize([128, 128], antialias=None)
transform1 = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.ToTensor(),
    ]
)


def process_input_image(img_path):
    image = plt.imread(img_path)
    image = image[:, :, 1]

    img = torch.from_numpy(image)
    img = img.unsqueeze(0)

    processed_img = transform(img)
    processed_img1 = transform1(processed_img)
    processed_img1 = processed_img1.unsqueeze(0)

    return processed_img1


def process_input_image2(image_file):
    image = Image.open(image_file)
    #image = np.array(image)
    image = np.array(image).astype(np.float32) / 255.0

    image = image[:, :, 1]

    img = torch.from_numpy(image).float()
    img = img.unsqueeze(0)

    processed_img = transform(img)
    processed_img1 = transform1(processed_img)
    processed_img1 = processed_img1.unsqueeze(0)

    return processed_img1


# segmentation part
def segment_image(input_image):
    processed_img = process_input_image(input_image)
    processed_img = processed_img.to(device)

    predicted_disc = generator_disc(processed_img)  # prediction
    predicted_disc1 = torch.round(predicted_disc)

    disc_img = predicted_disc1[0][0].detach().cpu().numpy()
    disc_img = np.round(disc_img)

    predicted_cup = generator_cup(processed_img)  # prediction
    predicted_cup1 = torch.round(predicted_cup)

    cup_img = predicted_cup1[0][0].detach().cpu().numpy()
    cup_img = np.round(cup_img)

    # Create directories if they don't exist
    os.makedirs("result", exist_ok=True)

    # Save disc image
    fig_disc, ax_disc = plt.subplots()
    ax_disc.imshow(disc_img)
    ax_disc.axis("off")
    disc_path = os.path.join("result", f"disc_seg_{os.path.basename(input_image)}")
    fig_disc.savefig(disc_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig_disc)

    # Save cup image
    fig_cup, ax_cup = plt.subplots()
    ax_cup.imshow(cup_img)
    ax_cup.axis("off")
    cup_path = os.path.join("result", f"cup_seg_{os.path.basename(input_image)}")
    fig_cup.savefig(cup_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig_cup)

    # Save combined image
    fig_combined, ax_combined = plt.subplots()
    ax_combined.imshow(disc_img)
    ax_combined.imshow(cup_img, cmap="jet", alpha=0.5 * (cup_img > 0))
    ax_combined.axis("off")
    img_path = os.path.join("result", f"full_seg_{os.path.basename(input_image)}")
    fig_combined.savefig(img_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig_combined)

    # Optionally show the combined image
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(disc_img)
    axes[0].axis("off")
    axes[1].imshow(cup_img)
    axes[1].axis("off")
    axes[2].imshow(disc_img)
    axes[2].imshow(cup_img, cmap="jet", alpha=0.5 * (cup_img > 0))
    axes[2].axis("off")
    # plt.show()

    return predicted_disc, predicted_cup


def segment_image2(input_image, filename):
    processed_img = process_input_image2(input_image)
    processed_img = processed_img.to(device)

    predicted_disc = generator_disc(processed_img)  # prediction
    predicted_disc1 = torch.round(predicted_disc)

    disc_img = predicted_disc1[0][0].detach().cpu().numpy()
    disc_img = np.round(disc_img)

    predicted_cup = generator_cup(processed_img)  # prediction
    predicted_cup1 = torch.round(predicted_cup)

    cup_img = predicted_cup1[0][0].detach().cpu().numpy()
    cup_img = np.round(cup_img)

    # Create directories if they don't exist
    os.makedirs("result", exist_ok=True)

    # Save disc image
    fig_disc, ax_disc = plt.subplots()
    ax_disc.imshow(disc_img)
    ax_disc.axis("off")
    disc_path = os.path.join("result", f"disc_seg_{filename}")  #
    fig_disc.savefig(disc_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig_disc)

    # Save cup image
    fig_cup, ax_cup = plt.subplots()
    ax_cup.imshow(cup_img)
    ax_cup.axis("off")
    cup_path = os.path.join("result", f"cup_seg_{filename}")  #
    fig_cup.savefig(cup_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig_cup)

    # Save combined image
    fig_combined, ax_combined = plt.subplots()
    ax_combined.imshow(disc_img)
    ax_combined.imshow(cup_img, cmap="jet", alpha=0.5 * (cup_img > 0))
    ax_combined.axis("off")
    img_path = os.path.join("result", f"full_seg_{filename}")  #
    fig_combined.savefig(img_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig_combined)

    # Optionally show the combined image
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(disc_img)
    axes[0].axis("off")
    axes[1].imshow(cup_img)
    axes[1].axis("off")
    axes[2].imshow(disc_img)
    axes[2].imshow(cup_img, cmap="jet", alpha=0.5 * (cup_img > 0))
    axes[2].axis("off")
    # plt.show()

    return predicted_disc, predicted_cup


# classification part
def classify_image(input_image):
    predicted_disc, predicted_cup = segment_image(input_image)

    pred_disc = torch.round(predicted_disc)
    pred_cup = torch.round(predicted_cup)

    overlap_image = pred_disc - pred_cup
    plt.imshow(overlap_image[0][0].detach().cpu().numpy())
    # plt.show()

    yhat3 = classifier3(overlap_image)
    yhat3 = yhat3.squeeze(1).item()

    print("glaucoma probability : ", yhat3)

    yhat = np.round(yhat3)
    if yhat == 0:
        print("no glaucoma")
    else:
        print("glaucoma")

    return yhat3, yhat


def classify_image2(input_image, filename):
    predicted_disc, predicted_cup = segment_image2(input_image, filename)

    pred_disc = torch.round(predicted_disc)
    pred_cup = torch.round(predicted_cup)

    overlap_image = pred_disc - pred_cup
    plt.imshow(overlap_image[0][0].detach().cpu().numpy())
    # plt.show()

    yhat3 = classifier3(overlap_image)
    yhat3 = yhat3.squeeze(1).item()

    print("glaucoma probability : ", yhat3)

    yhat = np.round(yhat3)
    if yhat == 0:
        print("no glaucoma")
    else:
        print("glaucoma")

    return yhat3, yhat


# execute
# input
if __name__ == "__main__":
    input_image = "data/images/drishtiGS_003.png"
    prob, galucoma = classify_image(input_image)