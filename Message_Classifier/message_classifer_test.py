from sklearn.externals import joblib
import pandas as pd
import chardet

class TestMessageClassifer():
	def __init__(self,modelpath='message_base_classifier.model'):
		self.classifier = joblib.load(modelpath)
		
	def batch_test(self,test_file_path,output_filename):
		with open(test_file_path, 'rb') as f:
			self.encode_result = chardet.detect(f.read()) 
			f.close()
		df = pd.read_csv(test_file_path,encoding=self.encode_result['encoding'])
		
		test_data=df['Message']
		predicted_lable=self.classifier.predict(test_data)
		df['Label']=predicted_lable
		print predicted_lable
		df[['RecordNo','Label']].to_csv(output_filename, index=False,encoding=self.encode_result['encoding'])
	
		
if __name__ == '__main__':
	messageclassifier= TestMessageClassifer()

	messageclassifier.batch_test('DEV_SMS.csv','LeaderBoard_DEV_SMS_label.csv')
	
	#Final Evaluation
	#messageclassifier.batch_test('TEST_SMS.csv','Evaluation_SMS_label.csv')