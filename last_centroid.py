# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to detect objects in a given image."""
from operator import itemgetter
import os
import numpy
from time import sleep
import psutil
import cv2
import argparse
import time

from PIL import Image
from PIL import ImageDraw

import detect
import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def convert(numbers):
    return int(''.join([ "%d"%x for x in numbers]))

def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}



def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])


def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-i', '--input', required=True,
                      help='File path of video to process.')
  parser.add_argument('-l', '--labels',
                      help='File path of labels file.')
  parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Score threshold for detected objects.')
  parser.add_argument('-o', '--output',
                      help='File path for the result image with annotations')
  parser.add_argument('-c', '--count', type=int, default=1,
                      help='Number of times to run inference')
  args = parser.parse_args()
  vidObj = cv2.VideoCapture(args.input) 
  # Used as counter variable 
  count = 0
  # checks whether frames were extracted 
  success = 1
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
  labels = load_labels(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()
  value = [[0]*20 for i in range(20)]
  obj_count = [0] * 20
  while success: 
  
        # vidObj object calls read 
        # function extract frames 
      
      success, image = vidObj.read()
      if(success == 1):   
        cv2.imwrite("frame%d.jpg" % count, image) 
        img = Image.open("frame%d.jpg" % count).convert('RGB') 
        scale = detect.set_input(interpreter, img.size,
                           lambda size: img.resize(size, Image.ANTIALIAS))
        print('----INFERENCE TIME----')
        for _ in range(args.count):
           start = time.perf_counter()
           interpreter.invoke()
           inference_time = time.perf_counter() - start
           objs = detect.get_output(interpreter, args.threshold, scale)
           print('%.2f ms' % (inference_time * 1000))

        print('-------RESULTS--------')
        frame_name = 'frame' + str(count) + '.jpg'
        os.remove(frame_name)
        fps = vidObj.get(cv2.CAP_PROP_FPS)
        print('fps')
        print(fps)
        loop_value = 0
        if not objs:
           print('No objects detected')

        for obj in objs:
          print(labels.get(obj.id, obj.id))
          print('  id:    ', obj.id)
          print('  score: ', obj.score)
          print('  bbox:  ', obj.bbox)
          print('  bbox:  ', obj.bbox.xmin)
          print('  bbox:  ', obj.bbox.ymin)
          print('  bbox:  ', obj.bbox.xmax)
          print('  bbox:  ', obj.bbox.ymax)
          centroid = [ (obj.bbox.xmin + obj.bbox.xmax) / 2 ,(obj.bbox.ymin + obj.bbox.ymax) / 2 ]
          value[count][loop_value] = centroid
          loop_value = loop_value + 1
          print(' centroid ', centroid)
          print(' value ',value)
          print('Frame count',count)
        obj_count[count] = loop_value
        for i in range(0,loop_value):
           for j in range(1,loop_value-1):
             if(value[count][i][0]>value[count][j][0]):
                 temp = value[count][i]
                 
        print('Total objects',obj_count[count])
        if(count > 0):
            if(obj_count[count] == obj_count[count-1]):
               print('There are same number of persons in this frame')
               for obj in value:
                  for object in range(0,obj_count[count]):
                     print(len(obj[count]))
            elif(obj_count[count] < obj_count[count-1]):
               print('1 person is missing in this frame')
            else:
               print('There is a new person in this frame')
        count = count + 1
        draw_objects(ImageDraw.Draw(img), objs, labels)
        img.save('/home/pi/outputs/output' + str(count-1) +'.jpeg')
      fps = vidObj.get(cv2.CAP_PROP_FPS)
      print('fps of the video is:') 
      print(fps)
  print('Video has been processed successfully')

if __name__ == '__main__':
  main()
