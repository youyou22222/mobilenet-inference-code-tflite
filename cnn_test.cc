#include"cnn.h"
#include"cnn_wrapper.h"
#include<vector>
#include<string>

CnnWrapper *CreateInstance(const char *model_file_name, const char *dictfile){
    CnnWrapper *pcnn = nullptr;
    try{
        pcnn = new CnnWrapper(model_file_name, dictfile);
    }catch(std::runtime_error &re){
        return nullptr;
    }

    return pcnn;
}


CnnWrapper::CnnWrapper(const char *model_file_name, const char *dictfile){
   
    std::string modelfile(model_file_name);
    std::string dict_file(dictfile);
    pCnn = new tflite4cnn::Cnn(modelfile, dict_file);

}

char* CnnWrapper::predict(float *img){
    if(pCnn){
        tflite4cnn::Cnn *cnn = std::static_cast<tflite4cnn::Cnn*>(pCnn);
        std::vector<std::string> res = cnn ->predict(img);
        std::string hanzi ="";
        for(int i = 0; i < res.size(); ++i)
            hanzi += res[i];

        char *hanzi_pred = new[hanzi.size()+1];
        strncpy(hanzi_pred, hanzi.c_str(), hanzi.size());
        return hanzi_pred;
    }
    return nullptr;
}
