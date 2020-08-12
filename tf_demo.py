import tensorflow as tf
import numpy as np
from selfTool import bert_modeling,data
from selfTool import file as fl
import optimization
import collections,re
from selfTool import deep_learn as dl
from tensorflow.contrib.layers.python.layers import batch_norm

# 定义参数。
flags = tf.app.flags

flags.DEFINE_string('ckpt_path',None,'model params path')
flags.DEFINE_boolean('train',False,'train model')
flags.DEFINE_boolean('eval',False,'evaluate model')
flags.DEFINE_boolean('predict',True,'predict input and write enter file')
flags.DEFINE_boolean('save',True,'save model that is supported tensorflow serving')
flags.DEFINE_string('export_path','serving_model','导出的可部署模型路径')
flags.DEFINE_integer('num_labels',15,'class number')#thu:10
flags.DEFINE_string('train_file','/home/wcs/item/textClass/fd_data/train.pkl','the data of train')
flags.DEFINE_string('eval_file','/home/wcs/item/textClass/fd_data/eval.pkl','the data of eval')
flags.DEFINE_string('predict_file','','the data of predict')
flags.DEFINE_integer('seq_len',32,'item len')#thu:32
flags.DEFINE_integer('train_num',180000,'train data number')#fd:8473。tt:304332。thu:180000
flags.DEFINE_integer('eval_num',10000,'eval data number')#fd:953。tt:76083。thu:10000
flags.DEFINE_integer('epoch',150,'total epoch steps')
flags.DEFINE_integer('batch',120,'')
flags.DEFINE_boolean('use_tpu',False,'')
flags.DEFINE_float('learn_rate',0.0005,'start learn rate')
flags.DEFINE_integer('vocab_size',3904,'vocab size') #fd:20609。tt:7375。thu:3904
flags.DEFINE_integer('numFilters',128,'cnn filter')
flags.DEFINE_integer('embeddingSize',200,'embedding len')
flags.DEFINE_float('l2RegLambda',0.2,'正则化项使用的系数')
flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_integer('save_checkpoints_steps',2000,'save')
flags.DEFINE_integer('log_step',1000,'loging info')
flags.DEFINE_string('output_dir','ckpt','cord')
flags.DEFINE_integer('iterations_per_loop',1000,'')

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


FLAGS = flags.FLAGS


def reader_file2() ->"reader data file.(自定义)":
    ds = fl.op_file('arg/ds.json',method='read')
    vocab = fl.op_file('arg/words_id.json',method='read')

    if FLAGS.train:
        dt = ds['train']
    elif FLAGS.eval:
        dt = ds['eval']
    else:
        dt = ds['test']

    xids = []
    yids = []

    for s in dt:
        nx = []
        for w in s[0]:
            try:
                nx.append(vocab.get(w,vocab['<unk>']))
            except:
                nx.append(vocab['<unk>'])

        k = [0 for i in range(FLAGS.seq_len - len(nx))]
        nk = nx + k
        xids.append(nk)
        yids.append(s[1])
        
    return xids,yids

def reader_file(vocab=None):
    if FLAGS.train:
        x_name = 'data/trainx_words.npy'
        y_name = 'data/trainy_label.npy'
    else:
        x_name = 'data/testx_words.npy'
        y_name = 'data/testy_label.npy'
    
    x = np.load(x_name)
    y = np.load(y_name)

    xids = [[vocab.get(j,vocab['<unk>']) for j in i] for i in x]
    xs = data.padding(xids,FLAGS.seq_len,pad=vocab['<pad>'])

    return xs,y

def build_input_fn(data_fn):
    xs,ys = data_fn()

    _num = FLAGS.train_num if FLAGS.train==True else FLAGS.eval_num
    def input_fn(params):
        #batch = params['batch_size']

        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":tf.constant(xs,shape=[_num,FLAGS.seq_len],dtype=tf.int32),
            "label_ids":tf.constant(ys,shape=[_num],dtype=tf.int32)
        })

        if FLAGS.train:
            ds = d.repeat(FLAGS.epoch)
            d = ds.shuffle(buffer_size=100)
        
        df = d.batch(batch_size=FLAGS.batch)

        labels = [i for i in range(1,FLAGS.num_labels + 1)]
        iter_rator = df.make_one_shot_iterator()
        dt = iter_rator.get_next()

        return dt,labels

    return input_fn

def create_model(labels,input_ids,is_training):
    #dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
    dropoutKeepProb = 0.7 if FLAGS.train==True else 1
    # 定义l2损失
    l2Loss = tf.constant(0.0)

    # 词嵌入层
    with tf.name_scope("embedding"):
        # 利用预训练的词向量初始化词嵌入矩阵
        emb = fl.op_file('data/embedding.pkl',method='read',model='pkl')
        #W = tf.Variable(tf.cast(emb, dtype=tf.float32, name="word2vec") ,name="W")
        W = tf.Variable(initial_value=tf.truncated_normal(shape=[FLAGS.vocab_size,FLAGS.embeddingSize],stddev=0.02,mean=0),name='W',dtype=tf.float32)
        # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
        embeddedWords = tf.nn.embedding_lookup(W, input_ids)
        # 卷积的输入是思维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度
        embeddedWordsExpanded = tf.expand_dims(embeddedWords, -1)

    # 创建卷积和池化层
    pooledOutputs = []
    # 有三种size的filter，3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
    # filterSize:2,3,4,5    numFIlters:120      embeddingSize:200
    filterSizes = (2,3,4,5)
    for i, filterSize in enumerate(filterSizes):
        with tf.name_scope("conv-maxpool-%s" % filterSize):
            # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
            # 初始化权重矩阵和偏置
            filterShape = [filterSize, FLAGS.embeddingSize, 1, FLAGS.numFilters]
            W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[FLAGS.numFilters]), name="b")
            conv = tf.nn.conv2d(
                    embeddedWordsExpanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                
            # relu函数的非线性映射
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                
            # 池化层，最大池化，池化是对卷积后的序列取一个最大值
            pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, FLAGS.seq_len - filterSize + 1, 1, 1],  # ksize shape: [batch, height, width, channels]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
            pooledOutputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中

    # 得到CNN网络的输出长度
    numFiltersTotal = FLAGS.numFilters * len(filterSizes)
        
    # 池化后的维度不变，按照最后的维度channel来concat
    hPool = tf.concat(pooledOutputs, 3)
        
    # 摊平成二维的数据输入到全连接层
    hPoolFlat = tf.reshape(hPool, [-1, numFiltersTotal])

    # dropout
    with tf.name_scope("dropout"):
        hDrop = tf.nn.dropout(hPoolFlat, dropoutKeepProb)
        hDrop = batch_norm(hDrop,decay=0.9,is_training=FLAGS.train)

    # 全连接层的输出
    with tf.name_scope("output"):
        outputW = tf.get_variable(
                "outputW",
                shape=[numFiltersTotal, FLAGS.num_labels],
                initializer=tf.contrib.layers.xavier_initializer())
        outputB= tf.Variable(tf.constant(0.1, shape=[FLAGS.num_labels]), name="outputB")
        l2Loss += tf.nn.l2_loss(outputW)
        l2Loss += tf.nn.l2_loss(outputB)
        logits = tf.nn.xw_plus_b(hDrop, outputW, outputB, name="logits")
        if FLAGS.num_labels == 1:
            predictions = tf.cast(tf.greater_equal(logits, 0.0), tf.int32, name="predictions")
        elif FLAGS.num_labels > 1:
            predictions = tf.argmax(logits, axis=-1, name="predictions")

        
    # 计算二元交叉熵损失
    with tf.name_scope("loss"):        
        if FLAGS.num_labels == 2:
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                            labels=tf.cast(tf.reshape(input_ids, [-1, 1]), 
                                                            dtype=tf.float32))
        elif FLAGS.num_labels > 2:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                
        loss = tf.reduce_mean(losses) + FLAGS.l2RegLambda * l2Loss
    
    return loss,predictions



def build_model_fn():
    def model_fn(features, labels, mode, params):
        input_ids = features['input_ids']
        label_ids = features['label_ids']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # 构建模型。
        (total_loss, probabilities) = create_model(input_ids=input_ids,
                                                is_training=is_training,
                                                labels=label_ids)
        
        tvars = tf.trainable_variables()

        scaffold_fn = None
        bert_ckpt = open('bert_variable.txt','r')
        bert_vars = bert_ckpt.read().split('\n')
        #bert_vars = [v[0:-2] for v in _bert_vars]

        # 检测bert模型文件，并剔除部分变量。
        """
        cp = FLAGS.ckpt_path if FLAGS.train==True else FLAGS.output_dir + '/model.ckpt-10401'
        if cp!=None:
            init_vars = tf.train.list_variables(cp)
            assignment_map = collections.OrderedDict()
            vs = init_vars if FLAGS.train==True else tvars
            for xx in vs:
                if FLAGS.train:
                    if xx[0] not in bert_vars or re.search('embeddings',xx[0])!=None or re.search('train_global_step',xx[0])!=None \
                        or re.search('DNN',xx[0])!=None or re.search('Loss',xx[0])!=None:
                        continue
                    else:
                        assignment_map[xx[0]] = xx[0]
                else:
                    assignment_map[xx.name[0:-2]] = xx.name[0:-2]

        if FLAGS.use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(FLAGS.ckpt_path,assignment_map)
            return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        elif cp!=None:
            # 恢复模型
            tf.train.init_from_checkpoint(cp, assignment_map)
        """
        #global_step = tf.train.get_or_create_global_step()
        num_warmup_steps = (FLAGS.train_num // FLAGS.batch) * FLAGS.epoch * 10
        output_spec = None
        g_step = tf.train.get_global_step()
        if mode==tf.estimator.ModeKeys.TRAIN:
            #train_op = optimization.create_optimizer(total_loss, FLAGS.learn_rate, global_step, num_warmup_steps, FLAGS.use_tpu)
            #loss_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CNN|Loss')
            
            #learn_rate = tf.train.exponential_decay(FLAGS.learn_rate,g_step,num_warmup_steps,0.0001)
            train_op = tf.train.AdamOptimizer(FLAGS.learn_rate).minimize(total_loss,global_step=g_step)#var_list=loss_vars

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                          loss=total_loss,
                                                          train_op=train_op,
                                                          scaffold_fn=scaffold_fn)
        
        elif mode==tf.estimator.ModeKeys.EVAL:
            def metric_fn(labels,logits):
                #predicts = tf.argmax(logits,axis=-1,output_type=tf.int32)
                accu = tf.metrics.accuracy(labels=labels,predictions=logits)
                return {"accuracy":accu}
            
            eval_metric = (metric_fn,[label_ids,probabilities])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                        loss=total_loss,
                                                        eval_metrics=eval_metric,
                                                        scaffold_fn=scaffold_fn)
        else:
            predict_out = {"probabilities": probabilities}
            _out = {"serving_default":tf.estimator.export.PredictOutput(predict_out)}
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                        predictions=predict_out,
                                                        scaffold_fn=scaffold_fn,
                                                        export_outputs=_out)
        
        return output_spec

    return model_fn


def serving_input_fn():
    # 定义模型的导出部分，导出后可用tensorflow serving部署。
    # 调用接口时模型的输入。
    input_x = tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.seq_len],name='predict_x')
    input_y = tf.placeholder(dtype=tf.int32,shape=[None],name='predict_y')
    batch = input_x.shape[0]

    features = [tf.feature_column.numeric_column(key='input_ids',shape=[32],dtype=tf.int64),
                tf.feature_column.numeric_column(key='label_ids',shape=[],dtype=tf.int64)]
    
    features2 = {"input_ids":input_x,"label_ids":input_y}

    v2 = tf.estimator.export.ServingInputReceiver(features2,{"texts":input_x,'labels':input_y})
    feature_parser = tf.feature_column.make_parse_example_spec(features)
    v = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_parser)
    #return v
    #return tf.contrib.learn.InputFnOps(features2, None, default_inputs=features2)
    return v2

def main(v):
    global tf

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    """
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    """
    session_config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        log_step_count_steps=FLAGS.log_step,
        session_config=session_config,
        tpu_config=tf.contrib.tpu.TPUConfig(
                    iterations_per_loop=FLAGS.iterations_per_loop,
                    num_shards=FLAGS.num_tpu_cores,
                    per_host_input_for_training=None))

    data_fn = reader_file2

    input_fn = build_input_fn(data_fn)
    model_fn = build_model_fn()

    estimator = tf.contrib.tpu.TPUEstimator(use_tpu=FLAGS.use_tpu,
                                            model_fn=model_fn,
                                            config=run_config,
                                            train_batch_size=FLAGS.batch)
    
    if FLAGS.train:
        estimator.train(input_fn=input_fn)
        """
        for k in sorted(result.keys()):
            tf.logging.info("{}={}".format(k,result[k]))
        """
    elif FLAGS.eval:
        eval_input_fn = build_input_fn(data_fn)
        result = estimator.evaluate(input_fn=eval_input_fn)
        for key in sorted(result.keys()):
            tf.logging.info("{}={}".format(key, str(result[key])))
    else:
        predict_input_fn = build_input_fn(data_fn)
        result = estimator.predict(input_fn=predict_input_fn)

        estimator.export_savedmodel(FLAGS.export_path,serving_input_fn)
        # 预测结果写入文件
        save_file = FLAGS.output_dir + '/predict.txt'
        with open(save_file,'w') as p:
            # 返回的result是全部结果，并不是按batch返回的
            for i,prediction in enumerate(result):
                probabilities = prediction["probabilities"] #这是单个值。
                output_line = str(i) + '\t' + str(probabilities) + "\n"
                p.write(output_line)
    

if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()


