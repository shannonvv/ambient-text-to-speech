import os
import numpy as np
import tkinter as tk
import requests
import threading
import csv
import tensorflow as tf
import certifi
from urllib.request import urlopen
import cv2
import random
from pydub import AudioSegment
import pygame


print(tf.__version__)
model = tf.keras.models.load_model('saved_model.h5')
tf.autograph.set_verbosity(0)


class Handwriting:
    def __init__(self):
        self.recordUrl = None
        self.apiUrl = None
        self.imageUrl = None
        self.paintingUrl = None
        self.inscriptionText = None
        self.artistName = None
        self.artDate = None
        self.artTitle = None
        self.thresholdLevel = None
        self.boxTop = None
        self.boxLeft = None
        self.boxHeight = None
        self.boxWidth = None

        self.y1 = None
        self.y2 = None
        self.x1 = None
        self.x2 = None
        self.angle = None
        self.invert = None
        self.cropped_text = None
        self.num = 0
        self.x_offset = 0
        self.y_offset = 0
        self.get_next_record()

    def get_next_record(self):
        # self.num = (self.num % 10) + 1
        self.get_art_data(6)

    def get_art_data(self, row_num):
        try:
            with open('artwork-data.csv', newline='') as csvfile:
                csvreader = csv.reader(csvfile)
                lines = list(csvreader)
                row = row_num - 1

                if row < len(lines):
                    fields = lines[row]

                    # Assign the cell data to variables
                    self.recordUrl, self.apiUrl, self.imageUrl, self.paintingUrl, self.inscriptionText, self.artistName, self.artDate, self.artTitle, self.thresholdLevel, self.boxLeft, self.boxTop, self.boxWidth, self.boxHeight, self.angle, self.invert = map(
                        str.strip, fields)

                    # Ensure numeric fields are parsed to integers
                    self.boxLeft, self.boxTop, self.boxWidth, self.boxHeight = map(float, [self.boxLeft, self.boxTop,
                                                                                           self.boxWidth, self.boxHeight])
                    self.process_image()
                else:
                    print("Row index out of range")
        except FileNotFoundError as e:
            print("Error opening file:", e)

    def process_image(self):
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
        os.environ["SSL_CERT_FILE"] = certifi.where()

        u = urlopen(self.paintingUrl)
        raw_data = u.read()
        u.close()

        resp = requests.get(self.paintingUrl, stream=True).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        self.y1 = self.boxTop
        self.y2 = self.boxTop + self.boxHeight
        self.x1 = self.boxLeft
        self.x2 = self.boxLeft + self.boxWidth

        cropped_image = image[int(self.y1):int(self.y2), int(self.x1):int(self.x2)]
        height, width = cropped_image.shape[:2]
        center = (width/2, height/2)
        # rotate_matrix = cv2.getRotationMatrix2D(center, int(self.angle), scale=1)
        rotate_matrix = cv2.getRotationMatrix2D(center, int(0), scale=1)
        image_rotate = cv2.warpAffine(cropped_image, rotate_matrix, (width, height))
        aspect_ratio = width/height

        # padded_image = np.zeros((444, 444, 3), dtype=np.uint8)
        padded_image = np.full((444, 444, 3), 255, dtype=np.uint8)

        if height > width:
            new_height = 444
            new_width = round(new_height * aspect_ratio)
            self.x_offset = int((444 - new_width)/2)
            self.y_offset = 0
        else:
            new_width = 444
            new_height = round(new_width / aspect_ratio)
            self.x_offset = 0
            self.y_offset = int((444 - new_height)/2)

        resized_image = cv2.resize(image_rotate, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        padded_image[self.y_offset:self.y_offset + new_height, self.x_offset:self.x_offset + new_width] = resized_image

        # downpoints = (new_width, new_height)
        # resize_down = cv2.resize(image_rotate, downpoints, interpolation=cv2.INTER_LINEAR)
        resize_down = cv2.resize(padded_image, (444, 444), interpolation=cv2.INTER_LINEAR)
        bw_image = cv2.cvtColor(resize_down, cv2.COLOR_BGR2GRAY)
        threshold_value = 100
        _, binary_image = cv2.threshold(bw_image, threshold_value, 255, cv2.THRESH_BINARY)
        # inverted_image = 255 - binary_image
        # self.cropped_text = inverted_image
        self.cropped_text = binary_image


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.x_offset = handwriting.x_offset
        self.y_offset = handwriting.y_offset
        self.screen_width = 444
        self.screen_height = 444
        self.image = None
        self.image_with_dots = None
        self.image_with_dots_original = None
        self.image_with_dots_updated = None
        self.image_with_dots_original_copy = None

        self.result = None
        self.current_word = None
        self.predicting = False
        self.width_increasing = True
        self.height_increasing = True
        self.overlap = None
        self.extracted_area = None

        self.circles = []
        self.drawn_circles = []
        self.borders = []
        self.unique_coordinates = []

        self.width = 20
        self.height = 20
        self.label_width = 25
        self.widths = []
        self.heights = []
        self.x_ranges = []
        self.extracted_areas = []
        self.width_flags = []
        self.height_flags = []
        self.word_list = []

        pygame.mixer.init()

        self.create_image()

    def generate_borders(self, num_items):
        borders = []
        if num_items < 2:
            raise ValueError("Number of items must be at least 2")

        for i in range(num_items):
            if i == 0:
                border = (0, self.x_ranges[1][0])
            elif i == num_items - 1:
                border = (self.x_ranges[i-1][1], self.screen_width)
            else:
                border = (self.x_ranges[i-1][1], self.x_ranges[i+1][0])
            borders.append(border)
        return borders

    def draw_rectangles(self, image):
        #print('unique coords: '+str(self.unique_coordinates))
        image[:] = 255
        image[:] = self.image_with_dots_original_copy

        for index, coordinates in enumerate(self.unique_coordinates):
            x, y = coordinates
            half_width = self.widths[index] // 2
            half_height = self.heights[index] // 2
            random_number = random.randint(2, 30)

            x_left = (max(0, x - half_width))
            x_right = (min(self.screen_width, x + half_width))
            self.x_ranges[index] = x_left, x_right

            # Check if the rectangle exceeds the screen boundaries
            borders = self.generate_borders(len(self.unique_coordinates))

            if self.x_ranges[index][0] == borders[index][0] or self.x_ranges[index][1] == borders[index][1]:
                self.width_flags[index] = False
                self.widths[index] -= 1

            elif self.widths[index] <= random_number:
                self.width_flags[index] = True
                self.widths[index] += 1

            else:
                if self.width_flags[index]:
                    self.widths[index] += 1

                else:
                    self.widths[index] -= 1

            if y - half_height < self.y_offset or y + half_height > self.screen_height - self.y_offset:
            #if y - half_height < self.label_width or y + half_height > self.screen_height - self.label_width:
                self.height_increasing = False
                self.height_flags[index] = False
                self.heights[index] -= 1
            else:
                if self.heights[index] <= 10:
                    self.height_increasing = True
                    self.height_flags[index] = True
                    self.heights[index] += 1
                elif self.height_flags[index]:
                    self.heights[index] += 1
                else:
                    self.heights[index] -= 1

            # Ensure the width doesn't go below 1
            self.widths[index] = max(5, self.widths[index])
            self.heights[index] = max(5, self.heights[index])

            half_width = self.widths[index] // 2
            half_height = self.heights[index] // 2

            # bounding box
            top_left = (max(0, x - half_width), max(0, y - half_height))
            bottom_right = (min(self.screen_width, x + half_width), min(self.screen_height, y + half_height))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), thickness=1)

            # label box
            label_top_left = (max(0, x - half_width), max(0, y - half_height) - self.label_width)
            label_bottom_right = (max(0, x - half_width) + self.label_width, max(0, y - half_height))
            cv2.rectangle(image, label_top_left, label_bottom_right, (0, 255, 0), thickness=cv2.FILLED)

            # label text
            if self.current_word is not None:
                text = self.current_word[index]
                font = cv2.FONT_HERSHEY_PLAIN
                scale_factor = 2
                font_scale = 1 * scale_factor
                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                text_x = (label_bottom_right[0] + label_top_left[0] - (text_size[0] * 2 - 3)) // 2
                text_y = (label_bottom_right[1] + label_top_left[1] + (text_size[1] * 2 + 3)) // 2
                cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 0, 0), 2, cv2.LINE_AA)

            self.extracted_area = self.extract_area_within_rectangle(image, top_left, bottom_right)
            self.extracted_areas[index] = self.extracted_area

    def extract_area_within_rectangle(self, image, top_left, bottom_right):
        extracted_area = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        return extracted_area

    def preprocess_image(self, image, desired_size=(56, 56)):
        # Calculate the aspect ratio of the original image
        h, w = image.shape[:2]
        aspect_ratio = w / h

        # Resize the image while preserving the aspect ratio
        if aspect_ratio > 1:  # Landscape orientation
            new_w = desired_size[0]
            new_h = int(new_w / aspect_ratio)

        else:
            # Portrait or square orientation
            new_h = desired_size[1]
            new_w = int(new_h * aspect_ratio)

        resized_image = cv2.resize(image, (new_w, new_h))

        # Add padding to the resized image to match the desired size
        top_pad = (desired_size[1] - new_h) // 2
        bottom_pad = desired_size[1] - new_h - top_pad
        left_pad = (desired_size[0] - new_w) // 2
        right_pad = desired_size[0] - new_w - left_pad

        padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT,
                                          value=(0, 0, 0))

        # Convert to grayscale
        grayscale_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)

        # Normalize the image
        normalized_image = grayscale_image / 255.0

        # Expand the dimensions to match the expected shape (None, 56, 56, 1)
        preprocessed_image = np.expand_dims(normalized_image, axis=-1)

        return preprocessed_image

    def speed_change(self, sound, speed=1.0):
        sample_rate = 24000
        # Manually override the frame_rate (samples per second)
        sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
            "frame_rate": int(sound.frame_rate * speed)
        })

        # convert the sound with altered frame rate to a standard frame rate
        # so that regular playback programs will work
        return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

    def text_to_speech_with_effects(self):
        if self.current_word is not None:
            alphabet_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                                's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
            files = []
            audio_durations = []
            concatenated_audio = None
            for index, letter in enumerate(self.current_word):
                if letter.lower() in alphabet_letters:
                    languages = ['en', 'es', 'fr', 'pt', 'zh-CN', 'zh-TW']
                    filename = 'letter_sounds/' + str(random.choice(languages)) + '/' + str(letter.lower()) + '.mp3'
                    # filename = 'letter_sounds/en/' + str(letter.lower()) + '.mp3'
                    sound = AudioSegment.from_mp3(filename)
                    duration_ms = len(sound)
                    audio_durations.append(duration_ms)
                    files.append(sound)

            # filtered_letters = [letter for letter in files if letter != 'U']
            # silence_filename = 'letter_sounds/silence.mp3'
            # silence = AudioSegment.from_mp3(silence_filename)

            for index, sound in enumerate(files):
                if len(files) < 0:
                    break
                if index == 0:
                    concatenated_audio = files[0]
                    # current_duration = len(concatenated_audio)

                else:
                    # random_end = random.randint(100, 3000)
                    # silent_clip = silence[0: random_end]
                    # concatenated_audio_gap = concatenated_audio.append(silent_clip)
                    current_duration = len(concatenated_audio)
                    # concatenated_audio = concatenated_audio_gap.append(files[index])
                    concatenated_audio = concatenated_audio.append(files[index], crossfade=min(current_duration * .8, audio_durations[index] * .8))

            if concatenated_audio is not None:
                speed = random.uniform(0.7, 1)
                slow_sound = self.speed_change(concatenated_audio, speed=speed)
                slow_sound.export("concatenated_audio.wav", format="wav")
                sound = pygame.mixer.Sound("concatenated_audio.wav")
                sound.play()

            # while pygame.mixer.get_busy():
                # random_float = random.uniform(0.0, 5.0)
                # pygame.time.Clock().tick(0)

    def create_image(self):
        cv2.namedWindow('Gibberish')
        handwriting = Handwriting()
        self.cropped_text = handwriting.cropped_text

        self.points = np.argwhere(self.cropped_text == 0)

        # copy to rewrite the canvas with each new rectangle
        self.image_with_dots_original = cv2.cvtColor(self.cropped_text.copy(), cv2.COLOR_GRAY2BGR)
        self.image_with_dots = cv2.cvtColor(self.cropped_text.copy(), cv2.COLOR_GRAY2BGR)

        height, width = self.cropped_text.shape[:2]
        self.image_with_dots_updated = np.full((height, width, 3), 255, dtype=np.uint8)

        # find coordinates of black pixels
        black_pixels = np.column_stack(np.where(self.image_with_dots == 0))

        # Draw dots on black pixels
        for i in range(0, len(black_pixels), 10):  # every 10 pixels
            x, y, _ = black_pixels[i]
            number = random.randint(0, 255)
            dot_color = (0, 0, 0)
            dot_size = random.randint(1, 3)
            cv2.circle(self.image_with_dots_updated, (y, x), dot_size, dot_color, -1)

        self.image_with_dots_original_copy = self.image_with_dots_updated.copy()

        # Callback function for mouse click event
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # self.get_prediction()
                # Draw a red dot at the clicked position
                circle = cv2.circle(self.image_with_dots_updated, (x, y), 3, (0, 0, 255), -1)
                self.circles.append((x, y))
                self.drawn_circles.append((x, y))
            self.unique_coordinates = list(set(self.circles))
            self.unique_coordinates = sorted(self.unique_coordinates, key=lambda coord: coord[0])

        # create a thread for predictions
        prediction_thread = threading.Thread(target=self.get_prediction)
        prediction_thread.daemon = True  # Daemonize the thread to allow it to terminate with the main thread
        prediction_thread.start()

        # Set the mouse callback function
        cv2.setMouseCallback('Gibberish', mouse_callback)

        while True:

            if self.predicting:
                self.draw_rectangles(self.image_with_dots_updated)

            cv2.imshow('Gibberish', self.image_with_dots_updated)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Press Esc key to exit
                break
            elif key == ord('g'):
                self.key_event(0)
            elif key == ord('c'):
                self.key_event(1)

        cv2.destroyAllWindows()

    def key_event(self, event):
        if event == 0:
            print("Predictions are starting")
            for i in range(0, len(self.unique_coordinates)):
                self.widths.append(random.randint(5, 20))
                self.heights.append(random.randint(5, 20))
                self.x_ranges.append((0, 0))
                self.extracted_areas.append(None)
                self.width_flags.append(True)
                self.height_flags.append(True)

            self.predicting = True

        elif event == 1:
            print("Dot has been drawn")

    def get_prediction(self):
        while True:
            if self.predicting:

                labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                          "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "d", "e",
                          "f", "g", "h", "n", "q", "r", "t"]

                letter = []

                for area in self.extracted_areas:
                    if area is not None:
                        preprocessed_image = self.preprocess_image(area)
                        prediction = model.predict(np.expand_dims(preprocessed_image, axis=0), verbose=0)
                        class_index = np.argmax(prediction)
                        confidence_score = prediction[0][class_index]
                        predicted_label = labels[class_index]
                        letter.append(predicted_label)

                if len(letter) == len(self.unique_coordinates) and len(letter) != 0:
                    self.current_word = letter
                    self.text_to_speech_with_effects()
                    self.result = ''.join(letter)
                    self.word_list.append(self.result)


if __name__ == "__main__":
    root = tk.Tk()
    handwriting = Handwriting()
    app = Application(root)
    app.mainloop()
