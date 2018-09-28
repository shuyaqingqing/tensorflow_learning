import tarfile
import tensorflow as tf
import os
import requests

inception_model_url="http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
inception_model_dir="inception_model"
if not os.path.exists(inception_model_dir):
    os.makedirs(inception_model_dir)

file_name=inception_model_url.split('/')[-1]
file_path=os.path.join(inception_model_dir,file_name)

if not os.path.exists(file_path):
    print("download: ",file_name)
    r=requests.get(inception_model_url,stream=True)
    print(r)
    with open(file_path,"wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print("finish:",file_name)

tarfile.open(file_path,"r:gz").extractall(inception_model_dir)

log_dir="log/inception_log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

inception_graph_file=os.path.join(inception_model_dir,"classify_image_graph_def.pb")
with tf.Session() as sess:
    with tf.gfile.FastGFile(inception_graph_file,"rb") as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,name="")
    write=tf.summary.FileWriter(log_dir,sess.graph)#log_dir文件所在目录，第二个参数是事件文件要记录的图
    write.close()







