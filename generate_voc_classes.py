#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 17:27:48 2018

@author: bobby
"""

classname = ["hair color - uncertainty", "hair color - brown", "hair color - orange", "hair color - gray", "hair color - white", "hair color - blue", "hair color - yellow", "hair color - black",
        "Shoe Color - Unsure", "Shoe Color - Stripes", "Shoe Color - Brown / Camel", "Shoe Color - Orange", "Shoe Color - Polka Dot", "Shoe Color - Gray", "Shoe Color - White", "Shoe Color-Pink", "Shoe Color-Purple", "Shoe Color-Red", "Shoe Color-Green", "Shoe Color-Flesh/Nude", "Shoe Color-Flower", "Shoes Color-blue", "shoe color-yellow", "shoe color-black",
        "Head Accessories - Uncertainty", "Head Accessories - Masks", "Head Accessories - Scarves", "Head Accessories - Sunglasses", "Head Accessories - Hats / Headscarves", "Head Accessories - Shawls" , "Head Accessories - None", "Head Accessories - Normal Glasses", "Head Accessories - Colored Glasses",
        "age layer - uncertainty", "age layer - middle age", "age layer - infants", "age layer - juvenile", "age layer - old age", "age layer - youth",
        "Emotion - Uncertainty", "Emotion - General mood", "Emotion - bad mood", "Emotion - good mood",
        "Top sleeves - Unsure", "Top sleeves - Medium and long sleeves", "Top sleeves - Sleeveless / Slings / Tube tops", "Top sleeves - Short sleeves", "Top sleeves - Long sleeves" ,
        "Crowd Attributes - Uncertainty", "Crowd Attributes - Middle Age People", "Crowd Attributes - Pragmatic Salary", "Crowd Attributes - Business Elite", "Crowd Attributes - Infants", "Crowd Attributes - Students", "Crowd Attributes - Home People", "Crowd Attributes - Fashionista", "Crowd Attributes - Seniors",
        "Top Color - Unsure", "Top Color - Stripes", "Top Color - Brown / Camel", "Top Color - Orange", "Top Color - Polka Dot", "Top Color - Grey", "Top Color - White ", "Top Color - Pink", "Top Color - Purple", "Top Color - Red", "Top Color - Green", "Top Color - Flesh / Nude", "Top Color - Color", "Tops Color - Blue", "Top Color - Yellow", "Top Color - Black",
        "Underwear Color - Unsure", "Bottom Color - Stripes", "Bottom Color - Brown / Camel", "Bottom Color - Orange", "Bottom Color - Polka Dot", "Bottom Color - Gray ", "Bottom Color - White", "Under Color - Pink", "Bottom Color - Purple", "Bottom Color - Red", "Bottom Color - Green", "Bottom Color - Flesh / Naked Color", "Bottom color - suit", "Bottom color - blue", "Bottom color - yellow", "Bottom color - black",
        "Tops Style - Uncertainty", "Tops Style - Casual", "Tops Style - Business", "Tops Style - Home", "Tops Style - Workwear", "Tops Style - Simple", "Tops Style - Sports",
        "Hairstyle - Unsure", "Hairstyle - Bald", "Hairstyle - Short Hair", "Hairstyle - Bald", "Hairstyle - Long Hair",
        "Bottom length - Uncertain", "Bottom length - mid-length skirt", "Bottom length - mid-length pants", "Bottom length - short skirt", "Bottom length - shorts", "Bottom length - long skirt", "bottom length - trousers",
        "Gender uncertainty", "gender woman", "gender man",
        "Bottom style - not sure", "Bottom style - casual", "Bottom style - business", "Bottom style - home", "Bottom style - overalls", "Bottom style - simple", " Bottom style - sports",
        "Handbags - Uncertainty", "Luggage Handheld - Shoulder Bag", "Luggage Handheld - Backpack", "Luggage Handheld - Plastic Bag / Shopping Bag", "Luggage Handheld - Stroller / Child", "Handbag Holder -Handbags", "Handbags - diagonal", "Handbags - No", "Handbags - Pockets", "Handbags - Luggage", "Handbags - Shopping Cart", "Handbags - Umbrellas" ]

with open('/home/bobby/work/dataset/bussiness100/voc_classes.txt', 'a') as f:
    f.writelines([i+'\n' for i in classname])