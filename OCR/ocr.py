# Maksymilian Wojnar

from itertools import product
from numpy.fft import fft2, ifft2
from numpy import rot90
from sklearn import svm
import numpy as np
import cv2
import sys


def load_image(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    if img.shape[1] > 2000:
        scale = 2000 / img.shape[0]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    elif img.shape[0] > 2000:
        scale = 2000 / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    return img


def perspective_correction(img):
    blurred = cv2.bilateralFilter(img, 20, 150, 150)
    blurred = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 5)
    blurred = cv2.GaussianBlur(blurred, (25, 25), 25)
    blurred[blurred > 20] = 255

    contours, hierarchy = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contours)

    quad = []
    eps = 5

    while len(quad) != 4:
        quad = np.float32(cv2.approxPolyDP(contours, eps, True))
        eps += 5

    points = [[0, 0], [0, h], [w, h], [w, 0]]
    rect = []

    for q in quad:
        dists = [(points[i][0] - q[0][0]) ** 2 + (points[i][1] - q[0][1]) ** 2 for i in range(len(points))]
        rect.append([points[np.argmin(dists)]])

    rect = np.array(rect, dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, rect)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)


def denoise_and_threshold(img):
    img = cv2.fastNlMeansDenoising(img, None, 9, 7, 21)
    img = cv2.bilateralFilter(img, 5, 20, 20)
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 7)


def crop_frame(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 5)

    rows_sum = np.sum(blurred, axis=1) / 255
    rows_peaks = np.column_stack(np.where(rows_sum > (img.shape[1] // 3)))
    if len(rows_peaks) >= 6:
        start, end = np.min(rows_peaks), np.max(rows_peaks)
        img = img[start + 20:end - 20]

    cols_sum = np.sum(blurred, axis=0) / 255
    cols_peaks = np.column_stack(np.where(cols_sum > (img.shape[0] // 3)))
    if len(cols_peaks) >= 6:
        start, end = np.min(cols_peaks), np.max(cols_peaks)
        img = img[:, start + 20:end - 20]

    return img


def skew_correction(img):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    best_angle, best_val = 0, 0

    for angle in range(-45, 46, 1):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, img.shape[::-1], flags=cv2.INTER_LINEAR)
        result = np.var(np.sum(rotated, axis=1))

        if result > best_val:
            best_val = result
            best_angle = angle

    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    return cv2.warpAffine(img, M, img.shape[::-1], flags=cv2.INTER_LINEAR)


def preprocess(img, framed):
    if framed:
        img = perspective_correction(img)
        img = denoise_and_threshold(img)
        img = crop_frame(img)
    else:
        img = denoise_and_threshold(img)

    img = skew_correction(img)
    return img


def find_intervals(points_sum, tr_val, padding):
    elements = []
    i = 0

    while i < len(points_sum):
        if points_sum[i] >= tr_val:
            start = i

            while i < len(points_sum) - 1 and points_sum[i] >= tr_val:
                i += 1

            end = i
            size = end - start

            if padding:
                start = max(0, int(start - size / 4))
                end = min(len(points_sum) - 1, int(end + size / 4))

            elements.append((start, end))

        i += 1

    return elements


def split_rows(source, blur_val=15):
    source = cv2.GaussianBlur(source, (blur_val, blur_val), blur_val)
    _, source = cv2.threshold(source, 40, 255, cv2.THRESH_BINARY)

    rows_sum = np.sum(source, axis=1) / 255
    tr_val = source.shape[1] / 20

    return find_intervals(rows_sum, tr_val, True)


kernel_motion_blur = np.zeros((13, 13))
kernel_motion_blur[6, :] = 1 / 13


def split_words(source):
    source = cv2.GaussianBlur(source, (5, 5), 5)
    source = cv2.filter2D(source, -1, kernel_motion_blur)

    cols_sum = np.sum(source, axis=0) / 255
    tr_val = source.shape[0] / 20

    return find_intervals(cols_sum, tr_val, False)


def split_chars(word):
    _, _, stats, _ = cv2.connectedComponentsWithStats(word, 8)
    positions = [[s[cv2.CC_STAT_LEFT], s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH]] for s in stats]
    positions.sort(key=lambda x: x[0])

    if len(positions) <= 1:
        return []

    stack = [positions[1]]
    for i in range(2, len(positions)):
        last = stack.pop()

        if last[1] > positions[i][0]:
            last[1] = max(positions[i][1], last[1])
            stack.append(last)
        else:
            stack.append(last)
            stack.append(positions[i])

    return stack


def load_dft_data():
    with open('texts/alphabet.txt', encoding='utf-8') as file:
        alphabet_text = file.read().split()

    alphabet_img = cv2.imread('images/alphabet.png', cv2.IMREAD_GRAYSCALE)

    rows = split_rows(alphabet_img)
    alphabet = []
    i = 0

    for r_start, r_end in rows:
        row = alphabet_img[r_start:r_end]
        chars = split_chars(row)

        for c_start, c_end in chars:
            char_img = row[:, c_start:c_end]
            alphabet.append((char_img, alphabet_text[i]))

            i += 1

    return alphabet


def img_to_text_dft(char, alphabet):
    best_match = [float('inf'), '']
    f_value = np.max(ifft2(fft2(char) * fft2(rot90(char, 2))).real)

    for char_img, char_text in alphabet:
        scale = char.shape[1] / char_img.shape[1]
        char_img = cv2.resize(char_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        correlation = ifft2(fft2(char) * fft2(rot90(char_img, 2), char.shape)).real
        result = np.abs(f_value - np.max(correlation))

        if result < best_match[0]:
            best_match[0] = result
            best_match[1] = char_text

    return best_match[1]


def standard_resize(img, size=90):
    if img.shape[1] > img.shape[0]:
        height = int(img.shape[0] * size / img.shape[1])
        img = cv2.resize(img, (size, height), interpolation=cv2.INTER_LINEAR)
    else:
        width = int(img.shape[1] * size / img.shape[0])
        img = cv2.resize(img, (width, size), interpolation=cv2.INTER_LINEAR)

    top_border = (size - img.shape[0]) // 2
    bottom_border = size - img.shape[0] - top_border
    left_border = (size - img.shape[1]) // 2
    right_border = size - img.shape[1] - left_border

    return cv2.copyMakeBorder(img, top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT, value=0)


def load_svm_data():
    with open('texts/alphabet_big.txt', encoding='utf-8') as file:
        alphabet_text = file.read().split()

    alphabet_img = cv2.imread('images/alphabet_big.png', cv2.IMREAD_GRAYSCALE)

    rows = split_rows(alphabet_img, 5)
    alphabet_big = []

    for r_start, r_end in rows:
        row = alphabet_img[r_start:r_end]
        chars = split_chars(row)

        for c_start, c_end in chars:
            char = row[:, c_start:c_end]
            char = standard_resize(char)
            alphabet_big.append(char.flatten())

    classifier = svm.SVC()
    classifier.fit(alphabet_big, alphabet_text)

    return classifier


def img_to_text_svm(char, classifier):
    char = standard_resize(char)
    return classifier.predict([char.flatten()])[0]


def ocr(image, img_to_text, **kwargs):
    rows = split_rows(image)
    text = ''

    for r_start, r_end in rows:
        row = image[r_start:r_end]
        words = split_words(row)

        for w_start, w_end in words:
            word = row[:, w_start:w_end]
            chars = split_chars(word)

            for c_start, c_end in chars:
                text += img_to_text(word[:, c_start:c_end], **kwargs)

            text += ' '
        text += '\n'

    return text


def levenshtein(original, ocr_text):
    delta = lambda x, y: 0 if x == y else 1
    n, m = len(original), len(ocr_text)

    edit = np.zeros((2, m + 1), dtype=np.int32)
    edit[0, :] = np.arange(m + 1)
    edit[1, 0] = 1

    for i, j in product(range(1, n + 1), range(1, m + 1)):
        edit[i % 2, j] = min(edit[(i - 1) % 2, j] + 1,
                             edit[i % 2, j - 1] + 1,
                             edit[(i - 1) % 2, j - 1] + delta(original[i - 1], ocr_text[j - 1]))

    return 1 - edit[n % 2, -1] / max(n, m)


def main():
    framed = '--no-frame' not in sys.argv
    if not framed:
        del sys.argv[sys.argv.index('--no-frame')]

    img = load_image(sys.argv[1])
    img = preprocess(img, framed)

    alphabet = load_dft_data()
    classifier = load_svm_data()

    text_dft = ocr(img, img_to_text_dft, alphabet=alphabet)
    text_svm = ocr(img, img_to_text_svm, classifier=classifier)

    if len(sys.argv) > 2:
        with open(sys.argv[2], encoding='utf-8') as file:
            text = file.read()

        print(f'OCR with DFT accuracy: {levenshtein(text, text_dft):.0%}')
        print(text_dft)

        print(f'\nOCR with SVM accuracy: {levenshtein(text, text_svm):.0%}')
        print(text_svm)
    else:
        print('OCR with DFT:')
        print(text_dft)

        print('\nOCR with SVM:')
        print(text_svm)


if __name__ == '__main__':
    main()
