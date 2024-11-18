import json
import nvtx

class Logger():
    """
    This class is a logger that saves all the experiment data after the training and evaluation process is done.
    """
    
    @nvtx.annotate("main->log_experiment_data->init_logger", color="gray")
    def __init__(self, jobname, folder_name, repetition):
        """
        Initialization of the logger.
        """

        self.folder_name = folder_name
        self.jobname = jobname
        self.repetition = repetition

    @nvtx.annotate("main->log_experiment_data->log_config", color="gray")
    def log_config(self, job_name, batch_size, epochs, learning_rate, momentum, weight_decay, gpu_number, tf_version, hvd_ranks=None):
        """
        Log the configuration used for the experiment.
        """

        config = {}
        config["job_name"] = str(job_name)
        config["repetition"] = self.repetition
        config["batch_size"] = batch_size
        config["epochs"] = epochs
        config["learning_rate"] = learning_rate
        config["momentum"] = momentum
        config["weight_decay"] = weight_decay
        config["gpu_number"] = gpu_number
        config["tf_version"] = str(tf_version)
        config["hvd_ranks"] = str(hvd_ranks)
        with open(self.folder_name+"/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    @nvtx.annotate("main->log_experiment_data->log_training_history", color="gray")
    def log_training_history(self, history):
        """
        Logs the training history, accuracy and loss.
        """

        acc = history.history["accuracy"]
        loss = history.history["loss"]
        val_acc = history.history["val_accuracy"]
        val_loss = history.history["val_loss"]
        f = open(self.folder_name+"/training_data.txt", "w")
        text = "["
        for i in range(len(acc)):
            text += str(acc[i])+","
        text = text[:-1]
        text += "]\n"
        f.write(text)
        text = "["
        for i in range(len(loss)):
            text += str(loss[i])+","
        text = text[:-1]
        text += "]\n"
        f.write(text)
        text = "["
        for i in range(len(val_acc)):
            text += str(val_acc[i])+","
        text = text[:-1]
        text += "]\n"
        f.write(text)
        text = "["
        for i in range(len(val_loss)):
            text += str(val_loss[i])+","
        text = text[:-1]
        text += "]\n"
        f.write(text)
        f.close()
