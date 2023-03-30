import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class NAT(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None):
        """
        args:
            root - /workspace/Mayank/dataset/train
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().nat_dir if root is None else root
        super().__init__('NAT', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()
        seq_ids = list(range(0, len(self.sequence_list)))

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'nat'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(s) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, s):
        object_meta = OrderedDict({'object_class_name': ''.join(filter(str.isalpha, s)),
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        return ['0000building1','0130car1_11','0139car18_3','0404person2_3','0421car5_5','0000car10_1','0130car1_12','0172building1_1','0404person2_4','0421car5_6','0000car10_2','0130car1_13','0172building1_2','0404person2_5','0421car5_7','0000car10_3','0130car1_14','0172building1_3','0404person2_6','0421car5_8','0000car10_4','0130car1_15','0172car1','0404person2_7','0421car5_9','0000car10_5','0130car1_16','0173bike1_1','0404person2_8','0421car6_1','0000car10_6','0130car1_17','0173bike1_2','0404person2_9','0421car6_2','0000car10_7','0130car1_1','0173bike1_3','0404person3_1','0421car6_3','0000car11_1','0130car1_2','0173bike1_4','0404person3_2','0421car6_4','0000car11_2','0130car1_3','0173bike1_5','0404person3_3','0421car6_5','0000car11_3','0130car1_4','0173building1_1','0404person3_4','0421car6_6','0000car11_4','0130car1_5','0173building1_2','0404person3_5','0421car6_7','0000car12_1','0130car1_6','0173building1_3','0404person4_1','0421car6_8','0000car12_2','0130car1_7','0173building1_4','0404person4_2','0421car7','0000car13_10_1','0130car1_8','0173building1_5','0404person4_3','0421car8','0000car13_10_2','0130car1_9','0173building1_6','0404person4_4','0421car9_1','0000car13_11','0130car3_10','0173building1_7','0404person4_5','0421car9_2','0000car13_12','0130car3_11','0173building1_8','0405building1_1','0421car9_3','0000car13_13','0130car3_12','0173building1_9','0405building1_2','0421car9_4','0000car13_14','0130car3_1','0173building2','0405building1_3','0421car9_5','0000car13_1','0130car3_2','0173building3_1','0405building1_4','0421car9_6','0000car13_2','0130car3_3','0173building3_2','0406building1_1','0421truck1_1','0000car13_3','0130car3_4','0173building3_3','0406building1_2','0421truck1_2','0000car13_4','0130car3_5','0173building3_4','0406building1_3','0421truck1_3','0000car13_5','0130car3_6','0173building3_5','0406building2_1','0421truck1_4','0000car13_6','0130car3_7','0173building3_6','0406building2_2','0421truck2_10','0000car13_7','0130car3_8','0173building3_7','0407building1_1','0421truck2_11','0000car13_8','0130car3_9','0173building3_8','0407building1_2','0421truck2_12','0000car13_9','0130truck1_10','0173building4_10','0407building1_3','0421truck2_13','0000car14_1','0130truck1_11','0173building4_11','0407building1_4','0421truck2_14','0000car14_2','0130truck1_12','0173building4_12','0408bus1_1','0421truck2_15','0000car14_3','0130truck1_13','0173building4_1','0408bus1_2','0421truck2_16','0000car14_4','0130truck1_14','0173building4_2','0408bus2_1','0421truck2_17','0000car15_1','0130truck1_15','0173building4_3','0408bus2_2','0421truck2_18','0000car15_2','0130truck1_1','0173building4_4','0408bus2_3','0421truck2_19','0000car15_3','0130truck1_2','0173building4_5','0408bus2_4','0421truck2_1','0000car15_4','0130truck1_3','0173building4_6','0408bus2_5','0421truck2_20','0000car16_1','0130truck1_4','0173building4_7','0408bus2_6','0421truck2_21','0000car16_2','0130truck1_5','0173building4_8','0408bus3_1','0421truck2_22','0000car16_3','0130truck1_6','0173building4_9','0408bus3_2','0421truck2_23','0000car16_4','0130truck1_7','0173car1_1','0408bus3_3','0421truck2_2','0000car16_5','0130truck1_8','0173car1_2','0408bus3_4','0421truck2_3','0000car17_1','0130truck1_9','0173car1_3','0408bus3_5','0421truck2_4','0000car17_2','0130truck2_10','0173car1_4','0408bus3_6','0421truck2_5','0000car17_3','0130truck2_1','0173car1_5','0408bus3_7','0421truck2_6','0000car17_4','0130truck2_2','0173car2','0408bus3_8','0421truck2_7','0000car17_5','0130truck2_3','0173car3_1','0408bus3_9','0421truck2_8','0000car18_10','0130truck2_4','0173car3_2','0408car1','0421truck2_9','0000car18_11','0130truck2_5','0173car3_3','0408car2','0425bike1_10','0000car18_12','0130truck2_6','0173person1_1','0408car3_1','0425bike1_11','0000car18_13','0130truck2_7','0173person1_2','0408car3_2','0425bike1_12','0000car18_1','0130truck2_8','0173person1_3','0408car3_3','0425bike1_13','0000car18_2','0130truck2_9','0173person1_4','0408car3_4','0425bike1_14','0000car18_3','0131person1_1','0173person2_1','0408car3_5','0425bike1_15','0000car18_4','0131person1_2','0173person2_2','0408car3_6','0425bike1_16','0000car18_5','0131person1_3','0173person2_3','0408car4_10_1','0425bike1_17','0000car18_6','0131person1_4','0173person2_4','0408car4_10_2','0425bike1_18','0000car18_7','0131ship1_10','0173person3','0408car4_10_3','0425bike1_19','0000car18_8','0131ship1_11','0173person4','0408car4_11','0425bike1_1','0000car18_9','0131ship1_1','0173person5','0408car4_1','0425bike1_20','0000car19_1','0131ship1_2','0175bike1_1','0408car4_2','0425bike1_21','0000car19_2','0131ship1_3','0175bike1_2','0408car4_3','0425bike1_22','0000car19_3','0131ship1_4','0175bike1_5','0408car4_4','0425bike1_2','0000car19_4','0131ship1_5','0175car1_1','0408car4_5','0425bike1_3','0000car19_5','0131ship1_6','0175car1_2','0408car4_6','0425bike1_4','0000car1_1','0131ship1_7','0175car1_3','0408car4_7','0425bike1_5','0000car1_2','0131ship1_8','0184car1','0408car4_8','0425bike1_6','0000car1_3','0131ship1_9','0184car2_1','0408car4_9','0425bike1_7','0000car20_10','0131ship2_10','0184car2_2','0408car5_10','0425bike1_8','0000car20_11','0131ship2_11','0184car2_3','0408car5_11','0425bike1_9','0000car20_12','0131ship2_1','0184car2_4','0408car5_12','0425car1_10','0000car20_13','0131ship2_2','0184car2_5','0408car5_13','0425car1_11','0000car20_14','0131ship2_3','0184car2_6','0408car5_14','0425car1_12','0000car20_1','0131ship2_4','0184car3_1','0408car5_15','0425car1_13','0000car20_2','0131ship2_5','0184car3_2','0408car5_16','0425car1_14','0000car20_3','0131ship2_6','0184car3_3','0408car5_17','0425car1_15','0000car20_4','0131ship2_7','0184landmark1_1','0408car5_18','0425car1_16','0000car20_5','0131ship2_8','0184landmark1_2','0408car5_19','0425car1_17','0000car20_6','0131ship2_9','0184landmark1_3','0408car5_1','0425car1_18','0000car20_7','0133bus1_1','0184landmark1_4','0408car5_20','0425car1_19','0000car20_8','0133bus1_2','0186building1_1','0408car5_21','0425car1_1','0000car20_9','0133bus1_3','0186building1_2','0408car5_22','0425car1_20','0000car21','0133bus1_4','0186building1_3','0408car5_23','0425car1_21','0000car22_10','0133bus1_5','0186building1_4','0408car5_24','0425car1_22','0000car22_1','0133bus1_6','0186building2_1','0408car5_25','0425car1_23','0000car22_2','0133bus1_7','0186building2_2','0408car5_26','0425car1_24','0000car22_3','0133bus1_8','0186building2_3','0408car5_2','0425car1_2','0000car22_4','0133car1_1','0186building2_4','0408car5_3','0425car1_3','0000car22_5','0133car1_2','0186building3','0408car5_4','0425car1_4','0000car22_6','0133car1_3','0186group1_1','0408car5_5','0425car1_5','0000car22_7','0133car1_4','0186group1_2','0408car5_6','0425car1_6','0000car22_8','0133car1_5','0186group1_3','0408car5_7','0425car1_7','0000car22_9','0133car1_6','0186person1','0408car5_8','0425car1_8','0000car23_10','0133car1_7','0187building1','0408car5_9','0425car1_9','0000car23_11','0133car1_8','0233car1','0408car6_1','0425car2_1','0000car23_12','0133car1_9','0233landmark1','0408car6_2','0425car2_2','0000car23_13','0133car2_10','0233net','0408car6_3','0425car2_3','0000car23_14','0133car2_1','0258car1','0408car6_4','0425car2_4','0000car23_15','0133car2_2','0258car2','0408motor1_1','0425car2_5','0000car23_16','0133car2_3','0258car3','0408motor1_2','0425car2_6','0000car23_1','0133car2_4','0260car1_1','0408motor1_3','0425court1_1','0000car23_2','0133car2_5','0260car1_2','0408motor1_4','0425court1_2','0000car23_3','0133car2_6','0260car1_3','0408person1_1','0425court1_3','0000car23_4','0133car2_7','0261building1','0408person1_2','0425court1_4','0000car23_5','0133car2_8','0264bike1_10','0408person1_3','0425court1_5','0000car23_6','0133car2_9','0264bike1_11','0408person2','0425court1_6','0000car23_7','0133truck1_1','0264bike1_12','0408truck2_1','0425person1_1','0000car23_8','0133truck1_2','0264bike1_13','0408truck2_2','0425person1_2','0000car23_9','0133truck1_3','0264bike1_14','0408truck2_3','0425person1_3','0000car24_10','0133truck1_4','0264bike1_15','0408truck_1','0425person1_4','0000car24_11','0133truck1_5','0264bike1_16','0408truck_2','0425person1_5','0000car24_12','0133truck1_6','0264bike1_17','0408truck_3','0426person1_1','0000car24_13','0133truck1_7','0264bike1_18','0408truck_4','0426person1_2','0000car24_14','0136car1_10','0264bike1_19','0408truck_5','0426person1_3','0000car24_15','0136car1_11','0264bike1_1','0408truck_6','0426person1_4','0000car24_16','0136car1_1','0264bike1_20','0409bridge1_10','0426person1_5','0000car24_1_1','0136car1_2','0264bike1_21','0409bridge1_11','0426person1_6','0000car24_1_2','0136car1_3_1','0264bike1_22','0409bridge1_12','0426person1_7','0000car24_1_3','0136car1_3_2','0264bike1_23','0409bridge1_13','0426person1_8','0000car24_2','0136car1_4','0264bike1_24','0409bridge1_14','0426person1_9','0000car24_3','0136car1_5','0264bike1_25','0409bridge1_15','0426person2_1','0000car24_4','0136car1_6','0264bike1_26','0409bridge1_16','0426person2_2','0000car24_5','0136car1_7','0264bike1_27','0409bridge1_17','0426person2_3','0000car24_6','0136car1_8','0264bike1_28','0409bridge1_18','0426person2_4','0000car24_7','0136car1_9','0264bike1_29','0409bridge1_19','0426person3_1','0000car24_8','0137car1_1','0264bike1_2','0409bridge1_1','0426person3_2','0000car24_9','0137car1_2','0264bike1_30','0409bridge1_20','0426person3_3','0000car25','0137car1_3','0264bike1_31','0409bridge1_21','0426person3_4','0000car26_1','0137car1_4','0264bike1_32','0409bridge1_22','0426person3_5','0000car26_2','0137car4_1','0264bike1_33','0409bridge1_2','0426person3_6','0000car26_3','0137car4_2','0264bike1_3','0409bridge1_3','0426person3_7','0000car26_4','0137car4_3_1','0264bike1_4','0409bridge1_4','0426person3_8','0000car26_5','0137car4_3_2','0264bike1_5','0409bridge1_5','0426person4_1','0000car26_6','0137car4_4','0264bike1_6','0409bridge1_6','0426person4_2','0000car26_7','0137dog1','0264bike1_7','0409bridge1_7','0426person4_3','0000car2','0137person1_1','0264bike1_8','0409bridge1_8','0426person4_4','0000car3','0137person1_2','0264bike1_9','0409bridge1_9','0426person4_5','0000car4','0137person1_3','0264car1','0409bridge2_1','0426person4_6','0000car5','0137person1_4','0264car2_1','0409bridge2_2','0426person5','0000car6_10','0137person2_1','0264car2_2','0409bridge2_3','0427bike1_1','0000car6_11','0137person2_2','0264car2_3','0409bridge2_4','0427bike1_2','0000car6_12','0137person2_3','0264car2_4','0409bridge2_5','0427bike1_3','0000car6_13','0137person2_4','0264car2_5','0409bridge2_6','0427bike1_4','0000car6_14_1','0138car10','0264car2_6','0409bridge2_7','0427bike1_5','0000car6_14_2','0138car11_1','0264car2_7','0409building1_10','0427car1','0000car6_15','0138car11_2','0264car3','0409building1_11','0427car2','0000car6_16','0138car11_3','0264car4','0409building1_12','0427car3','0000car6_1','0138car12_1','0264car5_1','0409building1_13','0427car4_1','0000car6_2','0138car12_2','0264car5_2','0409building1_14','0427car4_2','0000car6_3','0138car12_3','0264car5_3','0409building1_15','0427car4_3','0000car6_4','0138car13_1','0264court1','0409building1_1','0427car5_1','0000car6_5','0138car13_2','0264person1_1','0409building1_2','0427car5_2','0000car6_6','0138car13_3','0264person1_2','0409building1_3','0427car5_3','0000car6_7','0138car13_4','0264person1_3','0409building1_4','0427car5_4','0000car6_8','0138car13_5','0264person1_4','0409building1_5','0427car5_5','0000car6_9','0138car13_6','0264person2_1','0409building1_6','0427car5_6','0000car7_1','0138car13_7','0264person2_2','0409building1_7','0427car5_7','0000car7_2','0138car13_8','0264person2_3','0409building1_9','0427car6_1','0000car7_3','0138car13_9','0264person2_4','0417car1','0427car6_2','0000car7_4','0138car5_1','0264person2_5','0417car2','0427car6_3','0000car8','0138car5_2','0264person2_6','0417truck1','0427group1_3','0000car9_1','0138car5_3','0264person2_7','0417truck2_1','0427group1_5','0000car9_2','0138car5_4','0264person2_8','0417truck2_2','0427group1_6','0000car9_3','0138car5_5','0264person2_9','0417truck2_3','0427group1_7','0000car9_4','0138car5_6','0264person3_10','0417truck2_4','0427group1_8','0000car9_5','0138car7_1','0264person3_11','0418car1_1','0427group1_9','0000crane1_1','0138car7_2','0264person3_1','0418car1_2','0427group2','0000crane1_2','0138car8','0264person3_2','0418car1_3','0427landmark1','0000crane1_3','0138car9_1','0264person3_3','0419excavator1_1','0427landmark2','0000exvacator1_1','0138car9_2','0264person3_4','0419excavator1_2','0427motor1_1','0000exvacator1_2','0138car9_3','0264person3_5','0419excavator1_3','0427motor1_2','0000exvacator1_3','0138car9_4','0264person3_6','0419excavator_1','0427motor1_3','0000exvacator1_4','0138car9_5','0264person3_7','0419excavator_2','0427motor1_4','0000exvacator2_1','0138car9_6','0264person3_8','0419excavator_3','0427person1','0000exvacator2_2','0138dog101_1','0264person3_9','0419person1_1','0427runner1_10','0000group1_1','0138dog101_2','0264person4_1','0419person1_2','0427runner1_11','0000group1_2','0138dog101_3','0264person4_2','0419person1_3','0427runner1_12','0000group1_3','0138dog101_4','0264person4_3','0419person1_4','0427runner1_13','0000leaf1','0138dog101_5','0264person4_4','0419person1_5','0427runner1_14','0000lover2','0138electrombile2','0264person5_1','0419person1_6','0427runner1_15','0000lovers1_1','0138electrombile3_1','0264person5_2','0419person2_1','0427runner1_1','0000lovers1_2','0138electrombile3_2','0264person5_3','0419person2_2','0427runner1_2','0000lovers1_3','0138electrombile3_3','0264person5_4','0419person2_3','0427runner1_3','0000person1','0138electrombile3_4','0264person6','0419person2_4','0427runner1_4','0000person2_1','0138electrombile3_5','0264person7','0420car1_10','0427runner1_5','0000person2_2','0138person101','0264person8_1','0420car1_11','0427runner1_6','0000person2_3','0138person102','0264person8_2','0420car1_12','0427runner1_7','0000person3_1','0138person16_1','0264person8_3','0420car1_13','0427runner1_8','0000person3_2','0138person16_2','0264person8_4','0420car1_14','0427runner1_9','0000person3_3','0138person16_3','0264person8_5','0420car1_15','0427runner2_1','0000person5_10','0138person16_4','0264person8_6','0420car1_1','0427runner2_2','0000person5_11','0138person16_7','0265car1_1','0420car1_2','0427runner2_3','0000person5_12','0138person16_8','0265car1_2','0420car1_3','0427runner2_4','0000person5_1','0138person16_9','0265car1_3','0420car1_4','0427runner3','0000person5_2','0138person17_1','0266car2_1','0420car1_5','0427runner4_1','0000person5_3','0138person17_2','0266car2_2','0420car1_6','0427runner4_2','0000person5_4','0138person17_3','0266car2_3','0420car1_7','0427runner4_3','0000person5_5','0138person17_4','0267car3_10','0420car1_8','0427runner5_1','0000person5_6','0138person18_1','0267car3_11','0420car1_9','0427runner5_2','0000person5_7','0138person18_2','0267car3_12','0420car2','0427runner5_3','0000person5_8','0138person18_3','0267car3_13','0420car3','0427runner5_4','0000person5_9','0138person18_4','0267car3_14','0420forklift1','0427runner5_5','0000person6_10','0138person18_5','0267car3_15','0420forklift2_10','0427runner5_6','0000person6_11','0138person18_6','0267car3_16','0420forklift2_11','0427runner5_7','0000person6_12','0138person18_7','0267car3_17','0420forklift2_12','0427runner6_1','0000person6_13','0138person18_8','0267car3_18','0420forklift2_1','0427runner6_2','0000person6_14','0138person19_1','0267car3_19','0420forklift2_2','0427runner6_3','0000person6_15','0138person19_2','0267car3_1','0420forklift2_3','0427runner6_4','0000person6_16','0138person19_3','0267car3_20','0420forklift2_4','0427truck1_10','0000person6_17','0138person19_4','0267car3_21','0420forklift2_5','0427truck1_11','0000person6_18','0138person19_5','0267car3_22','0420forklift2_6','0427truck1_12','0000person6_19','0138person19_6','0267car3_23','0420forklift2_7','0427truck1_13','0000person6_1','0138person19_7','0267car3_24','0420forklift2_8','0427truck1_14','0000person6_20','0138person19_8','0267car3_2','0420forklift2_9','0427truck1_15','0000person6_21','0138person19_9','0267car3_3','0420motor1','0427truck1_1','0000person6_22','0138person4_10','0267car3_4','0420truck1_1','0427truck1_2','0000person6_2','0138person4_11','0267car3_5','0420truck1_2','0427truck1_3','0000person6_3','0138person4_12','0267car3_6','0420truck1_3','0427truck1_4','0000person6_4','0138person4_13','0267car3_7','0420truck2_1','0427truck1_5','0000person6_5','0138person4_14','0267car3_8','0420truck2_2','0427truck1_6','0000person6_6','0138person4_15','0267car3_9','0420truck2_3','0427truck1_7','0000person6_7','0138person4_16','0267car4_1','0420truck2_4','0427truck1_8','0000person6_8','0138person4_17','0267car4_2','0421car10_10','0427truck1_9','0000person6_9','0138person4_18','0267car4_3','0421car10_1','0427walker','0000person7_1','0138person4_19','0267group1_1','0421car10_2','0428car1_1','0000person7_2','0138person4_1','0267group1_2','0421car10_3','0428car1_2','0000person7_3','0138person4_20','0267group1_3','0421car10_4','0428car1_3','0000person7_4','0138person4_2','0267group1_4','0421car10_5','0428car1_4','0000person7_5','0138person4_3','0267group1_5','0421car10_6','0428car2','0000person7_6','0138person4_4','0267group1_6','0421car10_7','0428car3','0000person7_7','0138person4_5','0267person1_2','0421car10_8','0428car4_14','0000person7_8','0138person4_6','0267person1_3','0421car10_9','0428car4_15','0000person7_9','0138person4_7','0404car1_1_1','0421car11_10','0428car4_16','0000person8_1','0138person4_8','0404car1_1','0421car11_11','0428car4_17','0000person8_3','0138person4_9','0404car1_2','0421car11_12','0428car4_18','0000person8_4','0138person5_1','0404car1_3','0421car11_1','0428car4_19','0000rider1','0138person5_2','0404car1_4','0421car11_2','0428car4_1','0000rider2','0138person5_3','0404car1_5','0421car11_3','0428car4_21','0000rider3','0138person5_4','0404car1_6','0421car11_4','0428car4_2','0000rider4_10','0138person5_5','0404car2_1','0421car11_5','0428car4_3','0000rider4_1','0138person9_1','0404car2_2','0421car11_6','0428car4_4','0000rider4_2','0138person9_2','0404car2_3','0421car11_7','0428car4_5','0000rider4_3','0138person9_3','0404car2_4','0421car11_8','0428car4_6','0000rider4_4','0138person9_4','0404motor1_10','0421car11_9','0428car4_7','0000rider4_5','0138person9_5','0404motor1_11','0421car1','0428car4_8','0000rider4_6','0138tricycle1_10','0404motor1_12','0421car2','0428car4_9','0000rider4_7','0138tricycle1_11','0404motor1_13','0421car3_10','0428court1_1','0000rider4_8','0138tricycle1_12','0404motor1_14','0421car3_11','0428court1_2','0000rider4_9','0138tricycle1_13','0404motor1_15','0421car3_12','0428court1_3','0000rider5_1','0138tricycle1_14','0404motor1_16','0421car3_13','0428court1_4','0000rider5_2','0138tricycle1_15','0404motor1_17','0421car3_14','0428group1_10','0000rider5_3','0138tricycle1_16','0404motor1_1','0421car3_15','0428group1_11','0000rider5_4','0138tricycle1_1','0404motor1_2','0421car3_1','0428group1_1','0000rider6_1','0138tricycle1_2','0404motor1_3','0421car3_2','0428group1_2','0000rider6_2','0138tricycle1_3','0404motor1_4','0421car3_3','0428group1_3','0000rider6_3','0138tricycle1_4','0404motor1_5','0421car3_4','0428group1_4','0000runner1_1','0138tricycle1_5','0404motor1_6','0421car3_5','0428group1_5','0000runner1_2','0138tricycle1_6','0404motor1_7','0421car3_6','0428group1_6','0000runner1_3','0138tricycle1_7','0404motor1_8','0421car3_7','0428group1_7','0000runner1_4','0138tricycle1_8','0404motor1_9','0421car3_8','0428group1_9','0000runner2_1','0138tricycle1_9','0404motor2_1','0421car3_9','0428person1_1','0000runner2_2','0139car14_1','0404motor2_2','0421car4_10','0428person1_2','0000runner2_3','0139car14_2','0404motor2_3','0421car4_11','0428person2','0000runner2_4','0139car14_3','0404motor2_4','0421car4_12','0428person3_1','0000runner2_5','0139car15_1','0404person1_10','0421car4_13','0428person3_2','0000runner2_6','0139car15_2','0404person1_11','0421car4_1','0428person3_3','0000shoe1','0139car15_3','0404person1_12','0421car4_2','0428person3_4','0000sign1','0139car15_4','0404person1_13','0421car4_3','0428person3_5','0000sign2','0139car15_5','0404person1_1','0421car4_4','0428person3_6','0000sign3','0139car16_1','0404person1_2','0421car4_5','0428person3_7','0000sign4','0139car16_2','0404person1_4','0421car4_6','0428person3_8','0000sign5_1','0139car16_3','0404person1_5','0421car4_7','0428person4_1','0000sign5_2','0139car17_1','0404person1_7','0421car4_8','0428person4_2','0000sign5_3','0139car17_2','0404person1_8','0421car4_9','0428person4_3','0000streetlight1_1','0139car17_3','0404person1_9','0421car5_1','0428person4_4','0000streetlight1_2','0139car17_4','0404person2_10','0421car5_2','0000streetlight1_3','0139car18_1','0404person2_1','0421car5_3','0130car1_10','0139car18_2','0404person2_2','0421car5_4']

    def _read_bb_anno(self, bb_anno_file):
        # bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_values=-1,na_filter=True, low_memory=False).values
        # gt = gt.fillna(-1)
        return torch.tensor(gt)

    # def _read_target_visible(self, seq_path):
    #     raise NotImplementedError

    # def _get_sequence_path(self, seq_id):
    #     return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        anno_path = os.path.join(self.root, 'pseudo_anno', self.sequence_list[seq_id] + "_gt.txt")
        # print(anno_path)
        bbox = self._read_bb_anno(anno_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        # visible, visible_ratio = self._read_target_visible(seq_path)
        visible = valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': None}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:06}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        # seq_path = self._get_sequence_path(seq_id)
        # print("get_frame called")
        # print("Entered get frames")
        seq_path = os.path.join(self.root, 'NAT2021_train', 'train_clip', self.sequence_list[seq_id])
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        # print("reached here 0")
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        # print("reached here 1")
        if anno is None:
            # print("anno is None")
            anno = self.get_sequence_info(seq_id)
        # else:
        #     print(anno)
        # print("reached here 2")
        anno_frames = {}
        counter = 0
        for key, value in anno.items():
            print(counter, key)
            if(counter == 3):
                prin(value)
            counter += 1
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
            print(counter)
        print("Exit get frames")
        return frame_list, anno_frames, obj_meta
