import pickle
import tensorflow as tf
import cv2
import numpy as np
import time
import logging
import argparse

from tensorflow.python.client import timeline
from google.protobuf import json_format

from common import estimate_pose, CocoPairsRender, read_imgfile, CocoColors, draw_humans
from networks import get_network
from pose_dataset import CocoPoseLMDB

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s') #.INFO

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument('--imgpath', type=str, default='./images/jump2.png')
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--stage-level', type=int, default=6)
    parser.add_argument('--model', type=str, default='mobilenet', help='cmu / mobilenet / mobilenet_accurate / mobilenet_fast')
    args = parser.parse_args()

    input_node = tf.placeholder(tf.float32, shape=(1, args.input_height, args.input_width, 3), name='image')

    with tf.Session(config=config) as sess:

        net, _, last_layer = get_network(args.model, input_node, sess)

        logging.debug('read image+')
        image = read_imgfile(args.imgpath, args.input_width, args.input_height)
        # vec = sess.run(net.get_output(name='concat_stage7'), feed_dict={'image:0': [image]})

        a = time.time()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        pafMat, heatMat = sess.run(
            [
                net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
                net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
            ], feed_dict={'image:0': [image]}, options=run_options, run_metadata=run_metadata
        )
        logging.info('inference- elapsed_time={}'.format(time.time() - a))

        '''
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
        '''
        heatMat, pafMat = heatMat[0], pafMat[0]

        '''
        logging.debug('inference+')

        avg = 0
        rr = 10
        for _ in range(rr):
            a = time.time()
            sess.run(
                [
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
                ], feed_dict={'image:0': [image]}
            )
            logging.info('inference- elapsed_time={}'.format(time.time() - a))
            avg += time.time() - a
        logging.info('prediction avg= %f' % (avg / rr))
        '''

        '''
        logging.info('pickle data')
        with open('person3.pickle', 'wb') as pickle_file:
            pickle.dump(image, pickle_file, pickle.HIGHEST_PROTOCOL)
        with open('heatmat.pickle', 'wb') as pickle_file:
            pickle.dump(heatMat, pickle_file, pickle.HIGHEST_PROTOCOL)
        with open('pafmat.pickle', 'wb') as pickle_file:
            pickle.dump(pafMat, pickle_file, pickle.HIGHEST_PROTOCOL)
        '''

        '''
        logging.info('pose+')
        a = time.time()
        humans = estimate_pose(heatMat, pafMat)
        logging.info('pose- elapsed_time={}'.format(time.time() - a))
        # logging.debug(humans)

        logging.info('image={} heatMap={} pafMat={}'.format(image.shape, heatMat.shape, pafMat.shape))
        '''

        ''' # XX
        process_img = CocoPoseLMDB.display_image(image, heatMat, pafMat, as_numpy=True)

        # display
        image = cv2.imread(args.imgpath)
        image_h, image_w = image.shape[:2]
        image = draw_humans(image, humans)

        scale = 480.0 / image_h
        newh, neww = 480, int(scale * image_w + 0.5)

        image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

        convas = np.zeros([480, 640 + neww, 3], dtype=np.uint8)
        convas[:, :640] = process_img
        convas[:, 640:] = image
        
        # debug image and graph
        cv2.imshow('result', convas)
        cv2.waitKey(0)
        #tf.train.write_graph(sess.graph_def, '.', 'graph-tmp.pb', as_text=True)
        
        ''' # XX
        
        #tf.train.write_graph(sess.graph_def, '.', 'the-model.pb')
        
        # XX
        # get frozen model graph        
        # output_node_names =  [
        #     'Openpose/MConv_Stage6_L1_5_pointwise/BatchNorm/FusedBatchNorm',
        #     'Openpose/MConv_Stage6_L2_5_pointwise/BatchNorm/FusedBatchNorm'
        # ]
        # output_graph_def = tf.graph_util.convert_variables_to_constants(
        #     sess, # The session is used to retrieve the weights
        #     tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
        #     output_node_names # The output node names are used to select the usefull nodes
        # ) 
        # with tf.gfile.FastGFile('./the_model.pb', 'wb') as f: #GFile
        #     f.write(output_graph_def.SerializeToString()) # The graph_def is used to retrieve the nodes
        # print("%d ops in the final graph." % len(output_graph_def.node))
        
        # XX


        #new SavedModel (2018)
        # builder = tf.saved_model.builder.SavedModelBuilder('..\\tfjs\\saved_model')
        # builder.add_meta_graph_and_variables(sess,
        # [tf.saved_model.tag_constants.SERVING]) #TRAINING
        #builder.add_meta_graph(["bar-tag", "baz-tag"])
        # builder.save()

        tf.saved_model.simple_save(
            sess,
            'saved_model',
            inputs={"image": input_node},
            outputs={
                "pafMat": net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
                "heatMat": net.get_output(name=last_layer.format(stage=args.stage_level, aux=2)),
                })
        # THEN CONVERT WITH: tensorflowjs_converter --input_format=tf_saved_model --output_node_names=Openpose/MConv_Stage6_L1_5_pointwise/BatchNorm/FusedBatchNorm,Openpose/MConv_Stage6_L2_5_pointwise/BatchNorm/FusedBatchNorm --saved_model_tags=serve saved_model web_model
        
        '''
        json_string = json_format.MessageToJson(output_graph_def)
        with tf.gfile.FastGFile('./json_model.pb', 'wb') as fj: #GFile
            fj.write(json_string)
        '''

        '''
        # get the lite version of the graph
        out_tensors =  [
            net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
            net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
        ]
        tflite_model = tf.contrib.lite.toco_convert(output_graph_def, [input_node], out_tensors)
        open("converted_model.tflite", "wb").write(tflite_model)
        '''
                
                
#::::: OK ::::
#tensorflowjs_converter --input_format=tf_saved_model --output_node_names=Openpose/MConv_Stage6_L1_5_pointwise/BatchNorm/FusedBatchNorm,Openpose/MConv_Stage6_L2_5_pointwise/BatchNorm/FusedBatchNorm --saved_model_tags=serve saved_model web_model
#NEED A FIX: 
#C:\tes\tfjs\node_modules\@tensorflow\tfjs-converter\dist\data\compiled_api.d.ts  >>   import * as Long from "long"

