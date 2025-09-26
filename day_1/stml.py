from PIL import ImageFont, ImageDraw, Image
import numpy as np
from tqdm import tqdm


def STML(X, size=(224, 224), verbose=False, n_cols=None):
    n_features = X.shape[1]

    if n_cols == None:
        n_columns = 2 if n_features < 30 else 3
    else:
        n_columns = n_cols

    n_rows = np.ceil(n_features / n_columns)
    xs = np.ceil(size[0] / n_columns)
    ys = np.ceil(size[0] / n_rows)
    X = np.round(X, 4)
    # Coords for each text
    coords = []
    for i in range(int(n_rows)):
        for j in range(n_columns):
            coords.append((xs * j, ys * i))

    # Optmize font size
    X_string = list(map(str, X[:, :n_features].flatten().tolist()))
    longest_feature = max(X_string, key=len)

    max_font_size = 100
    for tmp_font_size in range(1, max_font_size):
        font = ImageFont.truetype("FreeSans.ttf", tmp_font_size)
        # feature_size = font.getsize(longest_feature)
        feature_size = font.getbbox(longest_feature)[2:4]
        # Stop if font is too large
        if feature_size[0] >= xs or feature_size[1] >= ys:
            break
        # A little smaller font size to ensure readability
        font_size = tmp_font_size - 2

    font = ImageFont.truetype("FreeSans.ttf", font_size)

    # Find longest string at a given font size for centering
    max_string_length = 100
    center_width = len(longest_feature)
    for string_length in range(len(longest_feature), max_string_length):
        tmp_longest = font.getbbox(''.join(" " for i in range(string_length + 1)))[2:4]
        if tmp_longest[0] >= xs:
            break
        center_width = string_length

    # Container for full set
    data = np.zeros((X.shape[0], size[0], size[1], 3))
    for sample in tqdm(range(X.shape[0]), disable=not verbose):
        img = np.zeros((size[0], size[1])).astype(np.uint8)

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        for id in range(n_features):
            draw.text((coords[id][0], coords[id][1]),
                      str(X[sample, id]),
                      font=font, fill="white")

        img = np.array(img)
        rgb = np.stack((img, img, img), axis=2)
        data[sample] = rgb
    return data.astype(np.uint8).squeeze()
