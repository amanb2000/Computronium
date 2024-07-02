import unittest
import sys 
import pdb 

import torch
import numpy as np
from tqdm import tqdm 

from computronium.video_utils import load_video


class TestUtils(unittest.TestCase): 
    @classmethod 
    def setUpClass(cls):
        """This is run once before all tests in this class. 
        """
        print("\n==================================")
        print("=== Setting up TestUtils class ===")
        print("==================================")


    def setUp(self): 
        """This is run before every single individual test method. 
        """
        ... 
    def tearDown(self): 
        """This is run after every single individual test method. 
        """
        ...


    #####################
    ### MESSAGE TESTS ###
    #####################


    def test_concatenation(self): 
        print("Testing `cat_msg_future()` function...")
