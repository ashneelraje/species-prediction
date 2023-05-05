import streamlit as st
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from keras.models import load_model
import sqlite3

model1 = load_model('./Model/abc.h5',compile=False)
lab={0: 'Africanized Bee (apis-mellifera)', 1: 'Algae (codium-fragile)', 2: 'American_Crow', 3: 'American_Goldfinch', 4: 'American_Three_toed_Woodpecker', 5: 'Anna_Hummingbird', 6: 'Armadillo (dasypus-novemcinctus)', 7: 'Artic_Tern', 8: 'Auklet', 9: 'Baby Turtle (chrysemys-picta)', 10: 'Baltimore_Oriole', 11: 'Bats (desmodus-rotundus)', 12: 'Belted_Kingfisher', 13: 'Beluga (delphinapterus-leucas)', 14: 'Bison(bison)', 15: 'Black_footed_Albatross', 16: 'Blue Whale (balaenoptera-musculus)', 17: 'Blue jay Bird (cyanocitta-cristata)', 18: 'Blue_Grosbeak', 19: 'Blue_winged_Warbler', 20: 'Boat_tailed_Grackle', 21: 'Bobolink', 22: 'Bohemian_Waxwing', 23: 'Brewer_Blackbird', 24: 'Brewer_Sparrow', 25: 'Broad-banded Copperhead Snake(agkistrodon-contortrix)', 26: 'Bronzed_Cowbird', 27: 'Brown_Pelican', 28: 'California_Gull', 29: 'Cape_Glossy_Starling', 30: 'Carolina_Wren', 31: 'Cattle (bos-taurus)', 32: 'Cerulean_Warbler', 33: 'Clark_Nutcracker', 34: 'Coelacanth (coelacanthiformes)', 35: 'Common_Raven', 36: 'Crested_Auklet', 37: 'Crocodile(crocodylus-niloticus)', 38: 'Dinosaur (ankylosaurus-magniventris)', 39: 'Dogs (canis-lupus-familiaris)', 40: 'Domestic Goose (branta-canadensis)', 41: 'Downy_Woodpecker', 42: 'Elegant_Tern', 43: 'Emperor Penguin(aptenodytes-forsteri)', 44: 'European_Goldfinch', 45: 'Fighting Fish (betta-splendens)', 46: 'Fish_Crow', 47: 'Fossa (cryptoprocta-ferox)', 48: 'Frigatebird', 49: 'Fruit Fly (ceratitis-capitata)', 50: 'Gadwall', 51: 'Gecko (correlophus-ciliatus)', 52: 'Geococcyx', 53: 'Golden Eagle (aquila-chrysaetos)', 54: 'Gray_Catbird', 55: 'Gray_Kingbird', 56: 'Great Blue heron( ardea-herodias)', 57: 'Harrier (circus-hudsonius)', 58: 'Heermann_Gull', 59: 'Indochinise Gaur(bos-gaurus)', 60: 'Jaguar', 61: 'Leopard', 62: 'Lion', 63: 'Macaw (ara-macao)', 64: 'Mallard', 65: 'Mallard Duck (anas-platyrhynchos)', 66: 'Megaladon (carcharodon-carcharias)', 67: 'Monarch Butterfly (danaus-plexippus)', 68: 'Moose(alces-alces)', 69: 'Nelson_Sharp_tailed_Sparrow', 70: 'Nighthawk', 71: 'Northern Cardinal (cardinalis-cardinalis)', 72: 'Northern Flicker (colaptes-auratus)', 73: 'Pacific_Loon', 74: 'Painted_Bunting', 75: 'Pigeon_Guillemot', 76: 'Pine_Grosbeak', 77: 'Pine_Warbler', 78: 'Poison Dart Frog (dendrobatidae)', 79: 'Purple_Finch', 80: 'Qinling Panda(ailuropoda-melanoleuca)', 81: 'Rattlesnake (crotalus-atrox)', 82: 'Red Panda (ailurus-fulgens)', 83: 'Red Squid (architeuthis-dux)', 84: 'Red-Eyed Tree Frog(agalychnis-callidryas)', 85: 'Red_headed_Woodpecker', 86: 'Rhinoceros (ceratotherium-simum)', 87: 'Ruby_throated_Hummingbird', 88: 'Scorpion (centruroides-vittatus)', 89: 'Sea Turtle (dermochelys-coriacea)', 90: 'Shiny_Cowbird', 91: 'Sloth (bradypus-variegatus)', 92: 'Smooth billed Ani (crotophaga-sulcirostris)', 93: 'Tiger', 94: 'Tortoise (centrochelys-sulcata)', 95: 'Tree_Sparrow', 96: 'Tree_Swallow', 97: 'Turtle (chelonia-mydas)', 98: 'Tyrannosaurus (diplodocus)', 99: 'Vulture (cathartes-aura)', 100: 'White_Pelican', 101: 'Wildebeest (connochaetes-gnou)', 102: 'Wolf (canis-lupus)', 103: 'Yellow_billed_Cuckoo', 104: 'cheetah(acinonyx-jubatus)'}



def processed_img(img_path):
    global x
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model1.predict(img)
    abcd = np.max(answer, axis=1)
    x= abcd
    if(abcd<=0.8):
        st.success('Unable to recognize.')
    else:
        y_class = answer.argmax(axis=-1)
        print(y_class)
        y = " ".join(str(x) for x in y_class)
        y = int(y)
        res = lab[y]
        print(res)
        return res


def run():
    st.title("Species Classification")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "Identify different species you find in the wild"</h4>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = 'D:/websites/species/images to predict/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())


        if st.button("Predict"):
            result = processed_img(save_image_path)
            if (x>=0.8):
                st.success("Predicted species is: "+result)
                conn = sqlite3.connect('Species.db')
                cursor = conn.cursor()
                c="select description from species where name='{}'".format(result)
                cursor.execute(c)
                answer = cursor.fetchone()
                ans=''.join(answer)
                st.success("Description: "+ ans)
                c = "select region from species where name='{}'".format(result)
                cursor.execute(c)
                answer = cursor.fetchone()
                ans = ''.join(answer)
                st.success("Region: " + ans)
                conn.commit()
                conn.close()

run()
