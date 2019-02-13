#include<sys/time.h>
#include<unordered_map>
#include<string>
#include<vector>
#include<iostream>
#include<fstream>
#include<utility>


#include"tensorflow/lite/kernels/register.h"
#include"tensorflow/lite/interpreter.h"
#include"tensorflow/lite/model.h"
#include"tensorflow/lite/string.h"

namespace tflite4mobile_net{
#define LOG(x) std::cerr
   class MobileNet{

        public:
           MobileNet(const std::string modelfile, const std::string dictfile);
           std::vector<std::pair<std::string, float>> predict(uint8_t *img, int topk=6);
           ~MobileNet(){
               if(buffer != nullptr)
                   delete buffer;
           }

        private:
           TfLiteStatus load_model_from_file(const std::string modelfile);
           TfLiteStatus load_model_from_buffer(const std::string buf);
           TfLiteStatus load_dict(const std::string dictfile);
           std::vector<size_t> argmax(const std::vector<float> &v);

           
        private:
           std::unique_ptr<tflite::FlatBufferModel> model_;
           std::unique_ptr<tflite::Interpreter> interpreter_;
           tflite::ops::builtin::BuiltinOpResolver resolver_;
           std::unordered_map<int, std::string>id2hanzi;

           char *buffer;
   };

}
