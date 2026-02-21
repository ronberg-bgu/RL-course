import unittest
import sys
import os

# Add the project root to the python path to import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.objects import SmallBox, BigBox

class TestCustomObjects(unittest.TestCase):
    
    def test_small_box_properties(self):
        box = SmallBox(color="blue")
        
        self.assertEqual(box.type, "box", "SmallBox should have base type 'box' for minigrid rendering.")
        self.assertEqual(box.color, "blue", "SmallBox color should be configurable.")
        self.assertEqual(box.box_size, "small", "SmallBox must have box_size='small'.")
        self.assertFalse(box.can_overlap(), "Agents should not be able to walk through boxes.")
        self.assertFalse(box.can_pickup(), "Agents should not be able to pick up boxes.")
        
    def test_big_box_properties(self):
        box = BigBox(color="red")
        
        self.assertEqual(box.type, "box", "BigBox should have base type 'box' for minigrid rendering.")
        self.assertEqual(box.color, "red", "BigBox color should be configurable.")
        self.assertEqual(box.box_size, "big", "BigBox must have box_size='big'.")
        self.assertFalse(box.can_overlap(), "Agents should not be able to walk through boxes.")
        self.assertFalse(box.can_pickup(), "Agents should not be able to pick up boxes.")

if __name__ == '__main__':
    unittest.main()
