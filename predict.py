import tensorflow as tf

def prediction():

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        train_loss, _, train_merged_sum = sess.run(
            [loss, optim, merged_sum], train_data_feed)


    print("Start training for stocks:", [d.stock_sym for d in dataset_list])
    for epoch in range(config.max_epoch):
        epoch_step = 0
        learning_rate = config.init_learning_rate * (
                config.learning_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
        )

        for label_, d_ in enumerate(dataset_list):
            for batch_X, batch_y in d_.generate_one_epoch(config.batch_size):
                global_step += 1
                epoch_step += 1
                batch_labels = np.array([[label_]] * len(batch_X))
                train_data_feed = {
                    self.learning_rate: learning_rate,
                    self.keep_prob: config.keep_prob,
                    self.inputs: batch_X,
                    self.targets: batch_y,
                    self.symbols: batch_labels,
                }
                train_loss, _, train_merged_sum = sess.run(
                    [self.loss, self.optim, self.merged_sum], train_data_feed)
                self.writer.add_summary(train_merged_sum, global_step=global_step)

                if np.mod(global_step, len(dataset_list) * 200 / config.input_size) == 1:
                    test_loss, test_pred = self.sess.run([self.loss_test, self.pred], test_data_feed)

                    # Plot samples
                    for sample_sym, indices in sample_indices.items():##iteritems():
                        image_path = os.path.join(self.model_plots_dir, "{}_epoch{:02d}_step{:04d}.png".format(
                            sample_sym, epoch, epoch_step))
                        sample_preds = test_pred[indices]
                        sample_truth = merged_test_y[indices]
                        self.plot_samples(sample_preds, sample_truth, image_path, stock_sym=sample_sym)

                    self.save(global_step)

    final_pred, final_loss = self.sess.run([self.pred, self.loss], test_data_feed)
