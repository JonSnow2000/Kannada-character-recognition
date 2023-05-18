from django.shortcuts import render
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2

preds = {
    0 : 'ಅ', 
    1 : 'ಆ',
    2 : 'ಇ',
    3 : 'ಈ',
    4 : 'ಉ',
    5 : 'ಊ',
    6 : 'ಋ',
    7 : 'ೠ',
    8 : 'ಎ',
    9 : 'ಏ',
    10 : 'ಐ',
    11 : 'ಒ',
    12 : 'ಓ',
    13 : 'ಔ',
    14 : 'ಅಂ',
    15 : 'ಅಃ',
    16 : 'ಕ',
    17 : 'ಖ',
    18 : 'ಗ',
    19 : 'ಘ',
    20 : 'ಙ',
    21 : 'ಚ',
    22 : 'ಛ',
    23 : 'ಜ',
    24 : 'ಝ', 
    25 : 'ಞ',
    26 : 'ಟ',
    27 : 'ಠ',
    28 : 'ಡ',
    29 : 'ಢ',
    30 : 'ಣ',
    31 : 'ತ',
    32 : 'ಥ',
    33 : 'ದ',
    34 : 'ಧ',
    35 : 'ನ',
    36 : 'ಪ',
    37 : 'ಫ',
    38 : 'ಬ',
    39 : 'ಭ',
    40 : 'ಮ',
    41 : 'ಯ',
    42 : 'ರ',
    43 : 'ಱ',
    44 : 'ಲ',
    45 : 'ಳ',
    46 : 'ೞ',
    47 : 'ವ',
    48 : 'ಶ',
    49 : 'ಷ',
    50 : 'ಸ',
    51 : 'ಹ',
    52 : '೦',
    53 : '೧',
    54 : '೨',
    55 : '೩',
    56 : '೪',
    57 : '೫',
    58 : '೬',
    59 : '೭',
    60 : '೮',
    61 : '೯'
}


def predict_fun(img):
    model = load_model('character_app/model.h5')
    img_pred = model.predict(img.reshape(-1, 64, 64, 1))
    return preds[np.argmax(img_pred)]

def predict_character(request):
    if request.method == 'POST':
        # Get the uploaded image from the request
        image_file = request.FILES['image']
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))  # Resize the image

        character = predict_fun(img)

        return render(request, 'result.html', {'character': character})

    return render(request, 'index.html')
