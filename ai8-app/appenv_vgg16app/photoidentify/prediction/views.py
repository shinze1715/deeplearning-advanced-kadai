from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from io import BytesIO
import os

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = img_array.reshape((1, 224, 224, 3))

            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)

            preds = model.predict(img_array)
            decoded_preds = decode_predictions(preds, top=5)[0]

            predictions = [(label, score) for _, label, score in decoded_preds]

            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form': form, 'predictions': predictions, 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})
        