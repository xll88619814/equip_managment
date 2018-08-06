# import the necessary packages
from skimage import feature

class HOG:
	def __init__(self, orientations = 9, pixelsPerCell = (8,8),
		cellsPerBlock = (3, 3), transform_sqrt = True, block_norm="L2"):
		# store the number of orientations, pixels per cell,
		# cells per block, and whether or not power law
		# compression should be applied
		self.orienations = orientations
		self.pixelsPerCell = pixelsPerCell
		self.cellsPerBlock = cellsPerBlock
                self.transform_sqrt = transform_sqrt
                self.block_norm = block_norm

	def describe(self, image):
		# compute HOG for the image
		hist = feature.hog(image, orientations = self.orienations,
			pixels_per_cell = self.pixelsPerCell,
			cells_per_block = self.cellsPerBlock,
                        transform_sqrt = self.transform_sqrt,
                        block_norm = self.block_norm)

		# return the HOG features
		return hist