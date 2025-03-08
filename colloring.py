import os
import tarfile
import tempfile
import urllib.request
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class DeepLabModel(object):
    """Class to load DeepLab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'

    def __init__(self, model_dir):
        """Creates and loads pretrained DeepLab model."""
        self.graph = tf.Graph()
        graph_def = None
        frozen_graph_path = os.path.join(model_dir, self.FROZEN_GRAPH_NAME)

        if not os.path.exists(frozen_graph_path):
            raise RuntimeError(f'Cannot find inference graph at {frozen_graph_path}')

        with tf.io.gfile.GFile(frozen_graph_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.Resampling.LANCZOS)

        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]}
        )
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

def download_and_load_model():
    """Downloads the model if not present and loads it."""
    model_name = 'deeplabv3_xception_ade20k_train'
    download_url = f'http://download.tensorflow.org/models/{model_name}_2018_05_29.tar.gz'
    model_dir = os.path.join(tempfile.gettempdir(), 'deeplab_model')
    saved_model_path = os.path.join(r"deeplab_model", 'saved_model')

    if os.path.exists(saved_model_path):
        print("Loading saved model...")
        return DeepLabModel(saved_model_path)

    # Download and extract model
    print("Downloading model, this might take a while...")
    os.makedirs(model_dir, exist_ok=True)
    tarball_path = os.path.join(model_dir, 'model.tar.gz')

    urllib.request.urlretrieve(download_url, tarball_path)
    print("Download completed! Extracting model...")

    with tarfile.open(tarball_path) as tar_file:
        tar_file.extractall(model_dir)

    # Rename extracted folder as saved_model for easy reuse
    extracted_folder = os.path.join(model_dir, model_name)
    if os.path.exists(extracted_folder):
        os.rename(extracted_folder, saved_model_path)

    print("Model extracted and saved successfully!")

    return DeepLabModel(saved_model_path)

# Load the model
# MODEL = download_and_load_model()
# print("Model is ready for inference!")

# orignal_im = '/image_path/image.png'
# resized_im, seg_map = MODEL.run(orignal_im)

label_names = {
    'wall': 1,
    'background': 2,
    'building, edifice': 3,
    'sky': 4,
    'floor, flooring': 5,
    'tree': 6,
    'ceiling': 7,
    'road, route': 8,
    'bed': 9,
    'windowpane, window ': 10,
    'grass': 11,
    'cabinet': 12,
    'sidewalk, pavement': 13,
    'person, individual, someone, somebody, mortal, soul': 14,
    'earth, ground': 15,
    'door, double door': 16,
    'table': 17,
    'mountain, mount': 18,
    'plant, flora, plant life': 19,
    'curtain, drape, drapery, mantle, pall': 20,
    'chair': 21,
    'car, auto, automobile, machine, motorcar': 22,
    'water': 23,
    'painting, picture': 24,
    'sofa, couch, lounge': 25,
    'shelf': 26,
    'house': 27,
    'sea': 28,
    'mirror': 29,
    'rug, carpet, carpeting': 30,
    'field': 31,
    'armchair': 32,
    'seat': 33,
    'fence, fencing': 34,
    'desk': 35,
    'rock, stone': 36,
    'wardrobe, closet, press': 37,
    'lamp': 38,
    'bathtub, bathing tub, bath, tub': 39,
    'railing, rail': 40,
    'cushion': 41,
    'base, pedestal, stand': 42,
    'box': 43,
    'column, pillar': 44,
    'signboard, sign': 45,
    'chest of drawers, chest, bureau, dresser': 46,
    'counter': 47,
    'sand': 48,
    'sink': 49,
    'skyscraper': 50,
    'fireplace, hearth, open fireplace': 51,
    'refrigerator, icebox': 52,
    'grandstand, covered stand': 53,
    'path': 54,
    'stairs, steps': 55,
    'runway': 56,
    'case, display case, showcase, vitrine': 57,
    'pool table, billiard table, snooker table': 58,
    'pillow': 59,
    'screen door, screen': 60,
    'stairway, staircase': 61,
    'river': 62,
    'bridge, span': 63,
    'bookcase': 64,
    'blind, screen': 65,
    'coffee table, cocktail table': 66,
    'toilet, can, commode, crapper, pot, potty, stool, throne': 67,
    'flower': 68,
    'book': 69,
    'hill': 70,
    'bench': 71,
    'countertop': 72,
    'stove, kitchen stove, range, kitchen range, cooking stove': 73,
    'palm, palm tree': 74,
    'kitchen island': 75,
    'computer, computing machine, computing device, data processor, electronic computer, information processing system': 76,
    'swivel chair': 77,
    'boat': 78,
    'bar': 79,
    'arcade machine': 80,
    'hovel, hut, hutch, shack, shanty': 81,
    'bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle': 82,
    'towel': 83,
    'light, light source': 84,
    'truck, motortruck': 85,
    'tower': 86,
    'chandelier, pendant, pendent': 87,
    'awning, sunshade, sunblind': 88,
    'streetlight, street lamp': 89,
    'booth, cubicle, stall, kiosk': 90,
    'television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box': 91,
    'airplane, aeroplane, plane': 92,
    'dirt track': 93,
    'apparel, wearing apparel, dress, clothes': 94,
    'pole': 95,
    'land, ground, soil': 96,
    'bannister, banister, balustrade, balusters, handrail': 97,
    'escalator, moving staircase, moving stairway': 98,
    'ottoman, pouf, pouffe, puff, hassock': 99,
    'bottle': 100,
    'buffet, counter, sideboard': 101,
    'poster, posting, placard, notice, bill, card': 102,
    'stage': 103,
    'van': 104,
    'ship': 105,
    'fountain': 106,
    'conveyer belt, conveyor belt, conveyer, conveyor, transporter': 107,
    'canopy': 108,
    'washer, automatic washer, washing machine': 109,
    'plaything, toy': 110,
    'swimming pool, swimming bath, natatorium': 111,
    'stool': 112,
    'barrel, cask': 113,
    'basket, handbasket': 114,
    'waterfall, falls': 115,
    'tent, collapsible shelter': 116,
    'bag': 117,
    'minibike, motorbike': 118,
    'cradle': 119,
    'oven': 120,
    'ball': 121,
    'food, solid food': 122,
    'step, stair': 123,
    'tank, storage tank': 124,
    'trade name, brand name, brand, marque': 125,
    'microwave, microwave oven': 126,
    'pot, flowerpot': 127,
    'animal, animate being, beast, brute, creature, fauna': 128,
    'bicycle, bike, wheel, cycle ': 129,
    'lake': 130,
    'dishwasher, dish washer, dishwashing machine': 131,
    'screen, silver screen, projection screen': 132,
    'blanket, cover': 133,
    'sculpture': 134,
    'hood, exhaust hood': 135,
    'sconce': 136,
    'vase': 137,
    'traffic light, traffic signal, stoplight': 138,
    'tray': 139,
    'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin': 140,
    'fan': 141,
    'pier, wharf, wharfage, dock': 142,
    'crt screen': 143,
    'plate': 144,
    'monitor, monitoring device': 145,
    'bulletin board, notice board': 146,
    'shower': 147,
    'radiator': 148,
    'glass, drinking glass': 149,
    'clock': 150,
    'flag': 151
}