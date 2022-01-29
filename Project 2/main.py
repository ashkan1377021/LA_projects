import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
from numpy.linalg import inv

def get_input(file_name):
    img = Image.open(file_name)
    img = np.asarray(img)
    img = to_mtx(img)
    return img


def to_mtx(img):
    """
    This method just reverse x and y of an image matrix because of the different order of x and y in PIL and Matplotlib library
    """
    H, V, C = img.shape
    mtr = np.zeros((V, H, C), dtype='int')
    for i in range(img.shape[0]):
        mtr[:, i] = img[i]
    return mtr


def get_coef(a, b, n):
    res = []
    b = [b[0], b[1], 1]
    dim = 3
    for i in range(dim):
        curr = [0] * dim * 4
        curr[i] = a[0]
        curr[dim + i] = a[1]
        curr[2 * dim + i] = 1 if i != 2 else 0

        curr[3 * dim + n - 1] = -b[i]
        res.append(curr)

    return res


def getPerspectiveTransform(pts1, pts2):
    A = []
    plen = len(pts1)

    for i in range(plen):
        A += get_coef(pts1[i], pts2[i], i)

    B = [0, 0, -1] * plen
    C = np.linalg.solve(A, B)

    res = np.ones(9)
    res[:8] = C.flatten()[:8]

    return res.reshape(3, -1).T


def showWarpPerspective(dst):
    width, height, _ = dst.shape

    # This part is for denoising the result matrix . You can use this if at first you have filled matrix with zeros
    for i in range(width - 1, -1, -1):
        for j in range(height - 1, -1, -1):
            if dst[i][j][0] == 0 and dst[i][j][1] == 0 and dst[i][j][2] == 0:
                if i + 1 < width and j - 1 >= 0:
                    dst[i][j] = dst[i + 1][j - 1]

    showImage(dst, title='Warp Perspective')


def showImage(image, title, save_file=True):
    final_ans = to_mtx(image)
    final_ans = final_ans.astype(np.uint8)

    plt.title(title)
    plt.imshow(final_ans)

    if save_file:
        try:
            os.mkdir('out')
        except OSError:
            pass
        path = os.path.join('out', title + '.jpg')
        plt.savefig(path, bbox_inches='tight')

    plt.show()


def Filter(img, filter_matrix):
    m, n, l = img.shape
    res = np.zeros((m, n, l))

    for i in range(m):
        for j in range(n):
            reshaped = np.reshape(img[i, j, :], newshape=(3,))
            res[i, j, :] = filter_matrix.dot(reshaped)

    return res



def warpPerspective(img, transform_matrix, output_width, output_height):
    """
    TODO : find warp perspective of image_matrix and return it
    :return a (width x height) warped image
    """
    new_img = np.zeros((output_width, output_height, 3))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            temp_coordinate = np.dot(transform_matrix, [x, y, 1])
            new_x = int(temp_coordinate[0] / temp_coordinate[2])
            new_y= int(temp_coordinate[1] / temp_coordinate[2])
            if new_x < output_width and new_y < output_height:
                new_img[new_x, new_y, :] = img[x, y, :]
    return new_img



def grayScaledFilter(img):
    """
    TODO : Complete this part based on the description in the manual!
    """
    filter_matrix = np.zeros((3,3))
    for i in range(3):
        filter_matrix[i] = [0.2,0.7,0.1]
    return Filter(img,filter_matrix)

def crazyFilter(img):
    """
    TODO : Complete this part based on the description in the manual!
    """
    x = np.zeros_like(img)
    imge = np.copy(img)
    imge[:,:,0] = img[:,:,1] +img[:,:, 2]
    imge[:, :, 1] = img[:, :, 0]
    imge[:,:,2] = x[:,:,2]
    return imge


def customFilter(img):
    """
    TODO : Complete this part based on the description in the manual!
    """
    filter_matrix = np.array([[0,1,0],[1,0,0],[0,0,1]])
    new_img = Filter(img,filter_matrix)
    showImage(new_img, title="custom Filter")
    inverse_filter_matrix = inv(np.matrix(filter_matrix))
    showImage(Filter(new_img,inverse_filter_matrix), title="Reverse")


def scaleImg(img, scale_width, scale_height):
    """
    TODO : Complete this part based on the description in the manual!
    """
    new_width = img.shape[0] * scale_width
    new_height = img.shape[1] * scale_height
    scaledImage = np.zeros((new_width, new_height, 3))
    for x in range(new_width):
        for y in range (new_height):
            new_x = int(x/scale_width)
            new_y= int(y/scale_height)
            scaledImage[x, y, :] = img[new_x, new_y, :]
    return scaledImage



def cropImg(img, start_row, end_row, start_column, end_column):
    """
    TODO : Complete this part based on the description in the manual!
    """
    croppedImage = np.zeros((end_column- start_column,end_row - start_row,3))
    for j in range(croppedImage.shape[0]):
        for i in range (croppedImage.shape[1]):
            croppedImage[j,i,:] = img[j+start_column , i+start_row,:]
    return croppedImage

if __name__ == "__main__":
    image_matrix = get_input('pic.jpg')

    # You can change width and height if you want
    width, height = 300, 400

    showImage(image_matrix, title="Input Image")

    # TODO : Find coordinates of four corners of your inner Image ( X,Y format)
    #  Order of coordinates: Upper Left, Upper Right, Down Left, Down Right
    pts1 = np.float32([[108, 216], [365, 177], [159, 643], [478, 574]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    m = getPerspectiveTransform(pts1, pts2)

    warpedImage = warpPerspective(image_matrix, m, width, height)
    showWarpPerspective( warpedImage)

    grayScalePic = grayScaledFilter(warpedImage)
    showImage(grayScalePic, title="Gray Scaled")

    crazyImage = crazyFilter(warpedImage)
    showImage(crazyImage, title="Crazy Filter")

    croppedImage = cropImg(warpedImage, 50, 300, 50, 225)
    showImage(croppedImage, title="Cropped Image")

    scaledImage = scaleImg(warpedImage, 2, 3)
    showImage(scaledImage, title="Scaled Image")

    customFilter(warpedImage)

