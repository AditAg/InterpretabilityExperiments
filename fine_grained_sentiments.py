import os
import subprocess
import pandas as pd
import numpy as np

class_path = ":/home/adit/Downloads/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/slf4j-api.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/jollyday.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/jaxb-core-2.3.0.1.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/ejml-0.23.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/javax.json-api-1.0-sources.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/slf4j-simple.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2-models.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2-javadoc.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2-sources.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/joda-time.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/joda-time-2.9-sources.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/jaxb-api-2.4.0-b180830.0359.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/xom-1.2.10-src.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/jollyday-0.4.9-sources.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/protobuf.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/javax.json.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/javax.activation-api-1.2.0-sources.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/jaxb-impl-2.4.0-b180830.0438-sources.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/xom.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/jaxb-impl-2.4.0-b180830.0438.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/jaxb-api-2.4.0-b180830.0359-sources.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/jaxb-core-2.3.0.1-sources.jar:/home/adit/Downloads/stanford-corenlp-full-2018-10-05/javax.activation-api-1.2.0.jar"



input_file = os.path.join(os.getcwd(), 'counterfactually-augmented-data')
input_file = os.path.join(input_file, 'sentiment')
input_file = os.path.join(input_file, 'new')
df_train = pd.read_csv(os.path.join(input_file, 'train.tsv'), sep = '\t')
data = df_train.values
data = data[1100:]
new_data = np.empty((data.shape[0], 3), dtype = object)
for i in range(data.shape[0]):
   print(i)
   new_data[i][0] = data[i][0]
   new_data[i][2] = data[i][1]
   process = subprocess.Popen(['java', '-cp', class_path, '-mx3g', 'edu.stanford.nlp.sentiment.SentimentPipeline', '-stdin'], stdin = subprocess.PIPE, stdout = subprocess.PIPE, universal_newlines = True, bufsize = 0)
   process.stdin.write(new_data[i][2])
   process.stdin.close()
   for line in process.stdout:
      if(new_data[i][1] == None):
         new_data[i][1] = str(line.strip())
      else:
         new_data[i][1] = new_data[i][1] + " " + str(line.strip())
      new_data[i][1] = str(new_data[i][1].strip())
   
   #process = subprocess.Popen(['java', '-cp', class_path, '-mx3g', 'edu.stanford.nlp.sentiment.SentimentPipeline', '-stdin'], stdin = subprocess.PIPE, stdout = subprocess.PIPE, universal_newlines = True)
   #output = process.communicate(input = new_data[i][1])
   #new_data[i][1] = output[0].strip()
   
cols = list(df_train.columns)
cols.insert(1, "New Sentiment")
new_df = pd.DataFrame(new_data, columns = cols)
new_df.to_csv(os.path.join(input_file, "new_train_2.tsv"), sep = '\t', index = None)
#while True:
#   output = process.stdout.readline()
#   print(output.strip())

