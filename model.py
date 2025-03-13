from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
def logo_company(image_path):

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    #print("Class:", class_name[2:], end="")
    #print("Confidence Score:", confidence_score)
    company_info = class_name[2:-1]
    descriptions = {
        "Google": "Google LLC — транснациональная корпорация из США в составе холдинга Alphabet, инвестирующая в интернет-поиск, облачные вычисления и рекламные технологии. Google поддерживает и разрабатывает многочисленные интернет-сервисы и продукты и получает прибыль в первую очередь от рекламы через свою программу Ads.",
        "Nvidia": "NVIDIA - мировой лидер в области визуальных вычислений. Графические процессоры, которые мы изобрели, выступают в роли «зрительной коры» современных компьютеров и лежат в основе всех наших продуктов.",
        "Apple": "Apple — американская корпорация, разработчик персональных и планшетных компьютеров, аудиоплееров, смартфонов, программного обеспечения и цифрового контента. Штаб-квартира расположена в Купертино, штат Калифорния.",
        "Microsoft": "Microsoft Corporation — американская публичная транснациональная корпорация, один из крупнейших в мире разработчиков в сфере проприетарного программного обеспечения для различного рода вычислительной техники — персональных компьютеров, игровых приставок, КПК, мобильных телефонов и прочего.",
        "PlayStation": "PlayStation Network (PSN) — платформа для скачивания цифрового мультимедиа и подключения к бесплатным и многопользовательским играм, предоставляемая Sony Computer Entertainment для консолей PlayStation 3, PlayStation 4, PlayStation 5, PlayStation Portable и PlayStation Vita. 4.",
        "Tesla": "Tesla, «Тесла» — американская компания, производитель электромобилей, зарядных станций и систем для хранения электроэнергии. Компания была основана в июле 2003 года Мартином Эберхардом и Марком Тарпеннингом, но нынешнее руководство компании называет сооснователями Илона Маска, Джеффри Брайана Страубела и Иэна Райта.",
        "Amazon": "Amazon — американская компания, крупнейшая в мире на рынках платформ электронной коммерции и публично-облачных вычислений по выручке и рыночной капитализации. Штаб-квартира — в Сиэтле.",
        "Xbox": "Xbox — серия игровых консолей от американской транснациональной корпорации Microsoft. Первая приставка вышла на рынок 15 ноября 2001 года, последняя на данный момент Xbox Series X/S — 10 ноября 2020 года.",
        "Samsung": "Samsung Electronics — транснациональная компания по производству полупроводников и электроники. Samsung Electronics является публичной компанией, акции которой торгуются на корейской и международных биржах. По состоянию на 2024 год 20 % акций компании принадлежит чеболю Samsung.",
        "Lego": "Lego — серии конструктора, представляющие собой наборы деталей для сборки и моделирования разнообразных предметов. Наборы Lego выпускает группа корпораций Lego Group, главный офис которой находится в Дании, в городе Биллунн. Компания была основана 10 августа 1932 года."
    }

    return descriptions.get(company_info, f"Неизвестная компания: {company_info}")
