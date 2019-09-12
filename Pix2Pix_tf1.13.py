import tensorflow as tf
import os
import warnings
import cv2
import numpy as np
from image_reader import generate_batch, file_path
from scipy.misc import imsave
warnings.simplefilter('ignore')


base_dir = './results'
LAMBDA = 100
EPS = 1e-12

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

model_dir = os.path.join(base_dir, 'Pix2Pix')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

checkpoint_dir = os.path.join(model_dir, "checkpoints")
checkpoint_save_path = os.path.join(checkpoint_dir, "my_model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

summary_dir = os.path.join(model_dir, "summaries")
if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)


def downsample(inp, filters, size, name, apply_batchnorm=True, pading='same'):
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):	
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.layers.conv2d(inp, filters, size, strides=2, padding=pading, kernel_initializer=initializer, use_bias=False)

		if apply_batchnorm:
			result = tf.layers.batch_normalization(result)

	result = tf.nn.leaky_relu(result)

	return result


def upsample(inp, concat_image, filters, size, name, apply_dropout=False, pading='same', apply_concat=True):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.layers.conv2d_transpose(inp, filters, size,
                                            strides=2,
                                            padding=pading,
                                            kernel_initializer=initializer, use_bias=False)

        result = tf.layers.batch_normalization(result)

        if apply_dropout:
            result = tf.nn.dropout(result, 0.5)

        result = tf.nn.relu(result)
        result = tf.concat([result, concat_image], axis=-1)
    return result


def Generator(input_image):
    #with tf.variable_scope('g_encoder'):
    down1 = downsample(input_image, 64, 4, 'g_down1', apply_batchnorm=False)
    down2 = downsample(down1, 128, 4, 'g_down2')
    down3 = downsample(down2, 256, 4, 'g_down3',pading='valid')
    down4 = downsample(down3, 512, 4, 'g_down4')
    down5 = downsample(down4, 512, 4, 'g_down5', pading='valid')
    down6 = downsample(down5, 512, 4, 'g_down6', pading='valid')
    down7 = downsample(down6, 512, 4, 'g_down7')
    down8 = downsample(down7, 512, 4, 'g_down8')

    with tf.variable_scope('g_decoder'):
    	up1 = upsample(down8, down7, 512, 4, 'g_up1', apply_dropout=True, apply_concat=False)
    	up2 = upsample(up1, down6, 512, 4, 'g_up2', apply_dropout=True)
    	up3 = upsample(up2, down5, 512, 4, 'g_up3', apply_dropout=True, pading='valid')
    	up4 = upsample(up3, down4, 512, 4, 'g_up4', pading='valid')
    	up5 = upsample(up4, down3, 256, 4, 'g_up5')
    	up6 = upsample(up5, down2, 128, 4, 'g_up6',pading='valid')
    	up7 = upsample(up6, down1, 64, 4, 'g_up7')
    with tf.variable_scope('g_last'):
        initializer = tf.random_normal_initializer(0., 0.02)
        final = tf.layers.conv2d_transpose(up7, 3, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation=tf.keras.activations.tanh)
        print(final.shape)
    return final


def d_downsample(inp, filters, size, stride, name, apply_batchnorm=True, pading='same'):
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):	
		initializer = tf.random_normal_initializer(0., 0.02)
		result = tf.layers.conv2d(inp, filters, size, strides=stride, padding=pading, kernel_initializer=initializer, use_bias=False)

		if apply_batchnorm:
			result = tf.layers.batch_normalization(result)

	result = tf.nn.leaky_relu(result)

	return result


def Discriminator(input_image, target_image):
	input_image = tf.concat([input_image, target_image], axis=-1)
	down1 = d_downsample(input_image, 64, 4, 2, 'd_down1', apply_batchnorm=False)
	down2 = d_downsample(down1, 128, 4, 2,'d_down2')
	down3 = d_downsample(down2, 256, 4, 1,'d_down3')

	
	with tf.variable_scope('d_last', reuse=tf.AUTO_REUSE) as scope:
		zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
		initializer = tf.random_normal_initializer(0., 0.02)
		conv1 = tf.layers.conv2d(zero_pad1, 512, 4, strides=1,
							    kernel_initializer=initializer)
	
		batchnorm1 = tf.layers.batch_normalization(conv1)
		leaky_relu = tf.nn.leaky_relu(batchnorm1)
		zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
		
		final = tf.layers.conv2d(zero_pad2, 1, 4,
                                 strides=1,
                                 kernel_initializer=initializer,
								 activation=tf.nn.sigmoid)
        
		print('final_shape', final.shape)
	
	return final


def discriminator_loss(disc_real_out, disc_generated_out):
	disc_loss = tf.reduce_mean(-(tf.log(disc_real_out + EPS) + tf.log(1 - disc_generated_out + EPS)))
	return disc_loss


def generator_loss(disc_generated_out, target, generated_image):
    gen_loss = tf.reduce_mean(-tf.log(disc_generated_out + EPS))
    L1_loss = tf.reduce_mean(tf.abs(target - generated_image))
    total_loss = L1_loss * LAMBDA + gen_loss
    return total_loss


#########################
#                       #
#   Creating The Graph  #
#                       #
#########################

tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('inputs'):
        target = tf.placeholder(tf.float32, shape=[None, 360, 360, 3])
        inpt = tf.placeholder(tf.float32, shape=[None, 360, 360, 3])

    gen_image = Generator(inpt)
    disc_real_out = Discriminator(inpt, target)
    disc_generated_out = Discriminator(inpt, gen_image)

    with tf.name_scope('loss'):
        disc_loss = discriminator_loss(disc_real_out, disc_generated_out)
        gen_loss = generator_loss(disc_generated_out, target, gen_image)
    
    with tf.name_scope('optimizers'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        generator_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(gen_loss, var_list=g_vars, global_step=global_step)
        discriminator_optimizer = tf.train.GradientDescentOptimizer(0.0002).minimize(disc_loss, var_list=d_vars, global_step=global_step)
    init = tf.global_variables_initializer()
    gen_loss_summary = tf.summary.scalar('Generator loss', gen_loss)
    disc_loss_summary = tf.summary.scalar('Discriminator loss', disc_loss)
    target_image_summary = tf.summary.image('Target image', target)
    gen_image_summary = tf.summary.image('Generated image', gen_image)
    input_image_summary = tf.summary.image('Input image', inpt)


def train(file_path, epochs):
    with tf.Session(graph=graph) as session:
        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
        saver = tf.train.Saver(max_to_keep=2)
        if checkpoint_state:
            print('Checkpoint is being restored.')
            checkpoint_path = checkpoint_state.model_checkpoint_path
            saver.restore(session, checkpoint_path)
        else:
            init.run(session=session)

        train_sw = tf.summary.FileWriter(os.path.join(summary_dir, 'train'), session.graph)

        for epoch in range(epochs):
            print('Epoch:', epoch)
            i = 1
            for input_batch, target_batch in generate_batch(file_path, 7):
                feed_dict = {target: target_batch, inpt: input_batch}
                
                (_,  gen_loss_sum, gen_image_sum,
                glob_step) = session.run([generator_optimizer,
                                          gen_loss_summary,
                                          gen_image_summary,
                                          global_step],
                                          feed_dict=feed_dict)

                (_, disc_loss_val, disc_loss_sum,
                target_image_sum, input_image_sum) = session.run([discriminator_optimizer,
                                                                  disc_loss,
                                                                  disc_loss_summary,
												                  target_image_summary,
                                                                  input_image_summary],
                                                                  feed_dict=feed_dict)

				
                (_, gen_loss_val, gen_loss_sum) = session.run([generator_optimizer,
                                                                  gen_loss,
                                                                  gen_loss_summary],
                                                                  feed_dict=feed_dict)
				
                train_sw.add_summary(gen_loss_sum, global_step=glob_step)
                train_sw.add_summary(disc_loss_sum, global_step=glob_step)
                train_sw.add_summary(target_image_sum, global_step=glob_step)
                train_sw.add_summary(gen_image_sum, global_step=glob_step)
                train_sw.add_summary(input_image_sum, global_step=glob_step)
                print('Epoch:', epoch, 'step:', i, 'Discriminator loss', disc_loss_val)
                if i >= 5:
                    input_image = cv2.imread(os.path.join(file_path, 'final_input', 'final_input{}.jpg'.format(str(i+15000))))
                    new_input = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                    normalized_input = new_input / 127.5 - 1
                    target_image = cv2.imread(os.path.join(file_path, 'final_target', 'final_target{}.jpg'.format(str(i+15000))))
                    feed_dict = {target: target_image[tf.newaxis, :], inpt: input_image[tf.newaxis, :]}
                    generated_image = session.run([gen_image], feed_dict=feed_dict)
                    generated_image = generated_image[0]
                    generated_image = ((generated_image + 1.0) * 127.0).astype(np.float32)
                    #print(np.array(generated_image).shape)
                    imsave(os.path.join(file_path, 'gen_images', 'gen_image{}.jpg'.format(str(i+15000))), generated_image[0])
                #if i % 100 == 0:n
                #    saver.save(session, checkpoint_save_path, global_step=glob_step)
                i += 1


train('./nikol_dataset', 40)


