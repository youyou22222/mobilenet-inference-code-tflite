#ifndef __CNN_WRAPPER_H
#define __CNN_WRAPPER_H
struct hanzi_prob{
    float *prob;
    char *hanzi;

};

class MobileNetWrapper{
    public:
        MobileNetWrapper(const char* model_file_name, const char *dictfile);
        struct hanzi_prob *predict(uint8_t *img, int topk=6);
        virtual ~MobileNetWrapper();
    private:
        void *pMobile;
    };

MobileNetWrapper *CreateInstance(const char* model_file_name, const char *dictfile);

#endif
