from pipeline.training_pipeline import training_pipeline

if __name__ == '__main__':
    training_pipeline(data_path='gs://zenml_quickstart/diabetes.csv')
    