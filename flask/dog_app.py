
class dog_breed_check(object):

    def __init__(self):
        from glob import glob
        from keras.applications.resnet50 import ResNet50
        from keras.applications.xception import Xception
        from keras.models import load_model
        import tensorflow as tf

        self.model = load_model('xception_dog_breed.h5')
        self.dog_names = self.get_dog_names()

        # define ResNet50 and Xception bottleneck model
        self.ResNet50_model = ResNet50(weights='imagenet')
        self.Xception_bottleneck = Xception(weights='imagenet', include_top=False)
        self.graph = tf.get_default_graph()


    # def create_model(self):
    #     from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
    #     from keras.layers import Dropout, Flatten, Dense
    #     from keras.models import Sequential

    #     ### TODO: Define your architecture.
    #     xception_model = Sequential()
    #     xception_model.add(GlobalAveragePooling2D(input_shape=[7, 7, 2048]))

    #     xception_model.add(Dropout(0.2))
    #     xception_model.add(Dense(133, activation='softmax'))

    #     xception_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #     ### TODO: Load the model weights with the best validation loss.
    #     xception_model.load_weights('saved_models/weights.best.xception.hdf5')

    #     return xception_model


    # Checks if there is a human or dog in the picture and finds the closest matching breed
    # -*- coding: UTF-8 -*-

    def check_breed(self, img_path):

        self.img_path = img_path
        gimage_path = ''
        with self.graph.as_default():
            isdog, ishuman = self.check_human_dog()
        
        # no dog or human detected
        if not isdog and not ishuman: 
            return "I can't see a dog or a human, try another picture!", False, ''

        # dog detected
        if isdog:
            # self.show_picture()
            breed = self.Xception_predict_breed()
            # print("I'm a %s! Woof!"%breed)
        # human detected
        elif ishuman:
            # self.show_picture()
            breed = self.Xception_predict_breed()
            # print("I'm no dog but I looke like a %s! \U0001f604"%breed)
            gimage_path = self.show_google_picture(breed[0])
        return breed, isdog, gimage_path


    ### TODO: Write a function that takes a path to an image as input
    ### and returns the dog breed that is predicted by the model.
    def Xception_predict_breed(self):
        import numpy as np   
        with self.graph.as_default(): 
            bottleneck_feature = self.extract_Xception(self.path_to_tensor())
            prediction = self.model.predict(bottleneck_feature)
        breeds = [self.dog_names[i] for i in self.mixed_breed(prediction)]
        return breeds


    def mixed_breed(self, prediction):
        import numpy as np
        highest = np.flip(prediction.argsort()[0][-2:],0)
        if prediction[0][highest[0]] > prediction[0][highest[1]]*1.5:
            return [highest[0]]
        else:        
            return highest


    def extract_Xception(self, tensor):
        from keras.applications.xception import preprocess_input
        return self.Xception_bottleneck.predict(preprocess_input(tensor))


    ### TODO: Write your algorithm.
    ### Feel free to use as many code cells as needed.

    # Function to check if human and/or dog is present in the image
    # returns dog,human with each either True or False
    def check_human_dog(self):
        dog = self.dog_detector()
        human = self.face_detector()
        return dog, human


        # returns "True" if face is detected in image stored at img_path
    def face_detector(self):  
        import cv2      
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
        img = cv2.imread(self.img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0


        ### returns "True" if a dog is detected in the image stored at img_path
    def dog_detector(self):
        prediction = self.ResNet50_predict_labels()
        return ((prediction <= 268) & (prediction >= 151)) 


    def ResNet50_predict_labels(self):
        from keras.applications.resnet50 import preprocess_input, decode_predictions
        import numpy as np
        # returns prediction vector for image located at img_path
        img = preprocess_input(self.path_to_tensor())
        return np.argmax(self.ResNet50_model.predict(img))


    def path_to_tensor(self):
        from keras.preprocessing import image                  
        from tqdm import tqdm
        import numpy as np
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(self.img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)


    # Saves the picture from the provided URL for future use
    def convert_url(self, url):
        from skimage import io
        import cv2
        from time import gmtime, strftime
        img = io.imread(url)
        img_path = 'input_'+strftime("%Y%m%d%H%M%S", gmtime())+'.jpg'
        cv2.imwrite('uploads/'+img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return img_path


    # Looks on google image search for a picture of the dog breed and displays it
    # The image search is a slightly modified version of this code: https://github.com/hardikvasa/google-images-download
    def show_google_picture(self, breed):
        img_path = self.google_search(breed)
        return img_path


    # Search for picture of dog on google images and display it
    # This bit is a modified version found on https://github.com/hardikvasa/google-images-download

    #Downloading entire Web Document (Raw Page Content)
    def download_page(self, url):
        import urllib.request    #urllib library for Extracting web pages
        try:
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
            req = urllib.request.Request(url, headers = headers)
            resp = urllib.request.urlopen(req)
            respData = str(resp.read())
            return respData
        except Exception as e:
            print(str(e))

    #Finding 'Next Image' from the given raw page
    def _images_get_next_item(self, s):
        start_line = s.find('rg_di')
        if start_line == -1:    #If no links are found then give an error!
            end_quote = 0
            link = "no_links"
            return link, end_quote
        else:
            start_line = s.find('"class="rg_meta"')
            start_content = s.find('"ou"',start_line+1)
            end_content = s.find(',"ow"',start_content+1)
            content_raw = str(s[start_content+6:end_content-1])
            return content_raw, end_content

    #Getting all links with the help of '_images_get_next_image'
    def _images_get_all_items(self, page):
        import time
        items = []
        j=0
        while j<5:
            item, end_content = self._images_get_next_item(page)
            if item == "no_links":
                break
            else:
                items.append(item)      #Append all the links in the list named 'Links'
                time.sleep(0.1)        #Timer could be used to slow down the request for image downloads
                page = page[end_content:]
            j+=1
        return items

    def google_search(self, search_keyword):
        from time import gmtime, strftime
        #Download Image Links
        search = search_keyword.replace(' ','%20')
        url = 'https://www.google.co.uk/search?q=' + search + '%20dog%20high%20resolution&site=webhp&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiU66C-ndzUAhWHtBQKHcWSDYkQ_AUIBigB'
        raw_html = (self.download_page(url))
        items = self._images_get_all_items(raw_html)

        k=0
        errorCount=0
        while(k<len(items)):
            from urllib.request import Request, urlopen    #urllib library for Extracting web pages
            from urllib.error import URLError, HTTPError

            try:
                req = Request(items[k], headers={"User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
                response = urlopen(req)
                img_path = 'gsearch_'+strftime("%Y%m%d%H%M%S", gmtime())+'.jpg'
                output_file = open("./uploads/"+img_path,'wb')
                data = response.read()
                output_file.write(data)
                response.close();
                k=k+1;
                break

            except IOError:   #If there is any IOError
                errorCount+=1
                k=k+1;
            except HTTPError as e:  #If there is any HTTPError
                errorCount+=1
                k=k+1;
            except URLError as e:
                errorCount+=1
                k=k+1;

        return img_path


    def get_dog_names(self):
        return ['Affenpinscher',
 'Afghan_hound',
 'Airedale_terrier',
 'Akita',
 'Alaskan_malamute',
 'American_eskimo_dog',
 'American_foxhound',
 'American_staffordshire_terrier',
 'American_water_spaniel',
 'Anatolian_shepherd_dog',
 'Australian_cattle_dog',
 'Australian_shepherd',
 'Australian_terrier',
 'Basenji',
 'Basset_hound',
 'Beagle',
 'Bearded_collie',
 'Beauceron',
 'Bedlington_terrier',
 'Belgian_malinois',
 'Belgian_sheepdog',
 'Belgian_tervuren',
 'Bernese_mountain_dog',
 'Bichon_frise',
 'Black_and_tan_coonhound',
 'Black_russian_terrier',
 'Bloodhound',
 'Bluetick_coonhound',
 'Border_collie',
 'Border_terrier',
 'Borzoi',
 'Boston_terrier',
 'Bouvier_des_flandres',
 'Boxer',
 'Boykin_spaniel',
 'Briard',
 'Brittany',
 'Brussels_griffon',
 'Bull_terrier',
 'Bulldog',
 'Bullmastiff',
 'Cairn_terrier',
 'Canaan_dog',
 'Cane_corso',
 'Cardigan_welsh_corgi',
 'Cavalier_king_charles_spaniel',
 'Chesapeake_bay_retriever',
 'Chihuahua',
 'Chinese_crested',
 'Chinese_shar-pei',
 'Chow_chow',
 'Clumber_spaniel',
 'Cocker_spaniel',
 'Collie',
 'Curly-coated_retriever',
 'Dachshund',
 'Dalmatian',
 'Dandie_dinmont_terrier',
 'Doberman_pinscher',
 'Dogue_de_bordeaux',
 'English_cocker_spaniel',
 'English_setter',
 'English_springer_spaniel',
 'English_toy_spaniel',
 'Entlebucher_mountain_dog',
 'Field_spaniel',
 'Finnish_spitz',
 'Flat-coated_retriever',
 'French_bulldog',
 'German_pinscher',
 'German_shepherd_dog',
 'German_shorthaired_pointer',
 'German_wirehaired_pointer',
 'Giant_schnauzer',
 'Glen_of_imaal_terrier',
 'Golden_retriever',
 'Gordon_setter',
 'Great_dane',
 'Great_pyrenees',
 'Greater_swiss_mountain_dog',
 'Greyhound',
 'Havanese',
 'Ibizan_hound',
 'Icelandic_sheepdog',
 'Irish_red_and_white_setter',
 'Irish_setter',
 'Irish_terrier',
 'Irish_water_spaniel',
 'Irish_wolfhound',
 'Italian_greyhound',
 'Japanese_chin',
 'Keeshond',
 'Kerry_blue_terrier',
 'Komondor',
 'Kuvasz',
 'Labrador_retriever',
 'Lakeland_terrier',
 'Leonberger',
 'Lhasa_apso',
 'Lowchen',
 'Maltese',
 'Manchester_terrier',
 'Mastiff',
 'Miniature_schnauzer',
 'Neapolitan_mastiff',
 'Newfoundland',
 'Norfolk_terrier',
 'Norwegian_buhund',
 'Norwegian_elkhound',
 'Norwegian_lundehund',
 'Norwich_terrier',
 'Nova_scotia_duck_tolling_retriever',
 'Old_english_sheepdog',
 'Otterhound',
 'Papillon',
 'Parson_russell_terrier',
 'Pekingese',
 'Pembroke_welsh_corgi',
 'Petit_basset_griffon_vendeen',
 'Pharaoh_hound',
 'Plott',
 'Pointer',
 'Pomeranian',
 'Poodle',
 'Portuguese_water_dog',
 'Saint_bernard',
 'Silky_terrier',
 'Smooth_fox_terrier',
 'Tibetan_mastiff',
 'Welsh_springer_spaniel',
 'Wirehaired_pointing_griffon',
 'Xoloitzcuintli',
 'Yorkshire_terrier']


if __name__ == "__main__":

    img_path = '../images/dog1.jpg'
    dog = dog_breed_check()
    dog.show_picture(img_path)
    breed, isdog, gimage = dog.check_breed(img_path)

    if isdog:
        print("I'm a %s! Woof!"%breed)
    else:
        print("I'm no dog but I looke like a %s! \U0001f604"%breed)
        dog.show_picture(gimage) 




