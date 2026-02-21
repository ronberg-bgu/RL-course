from minigrid.core.world_object import WorldObj

class SmallBox(WorldObj):
    """
    A small box that can be pushed by a single agent.
    """
    def __init__(self, color="yellow"):
        super().__init__("box", color)
        # Custom property to differentiate from regular minigrid boxes
        self.box_size = "small"

    def can_overlap(self):
        """Objects can't overlap with the box"""
        return False
        
    def can_pickup(self):
        """Boxes cannot be picked up, only pushed"""
        return False

class BigBox(WorldObj):
    """
    A big box that requires two agents pushing in the exact same direction simultaneously to move.
    """
    def __init__(self, color="purple"):
        # We can use a different type name or the same "box" type but differentiate via our property.
        # MiniGrid rendering uses the type string to pick the drawing method. "box" will draw a box.
        super().__init__("box", color)
        self.box_size = "big"

    def can_overlap(self):
        """Objects can't overlap with the box"""
        return False
        
    def can_pickup(self):
        """Boxes cannot be picked up, only pushed"""
        return False
