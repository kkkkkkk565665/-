#include <iostream>
#include "unistd.h"
#include "sdk.h"
#include "sample_media_ai.h"

using namespace std;

/* function : show usage */
static void SAMPLE_AI_Usage(char* pchPrgName)
{
    printf("Usage : %s <index> \n", pchPrgName);
    printf("index:\n");
    printf("\t 0) demo to open camera.\n");
}

int main(int argc, char *argv[])
{
    HI_S32 s32Ret = HI_FAILURE;
    if (argc < 2 || argc > 2) { // 2: argc indicates the number of parameters
        SAMPLE_AI_Usage(argv[0]);
        return HI_FAILURE;
    }

    if (!strncmp(argv[1], "-h", 2)) { // 2: used to compare the first 2 characters
        SAMPLE_AI_Usage(argv[0]);
        return HI_SUCCESS;
    }
    sdk_init();
    /* MIPI is GPIO55, Turn on the backlight of the LCD screen */
    system("cd /sys/class/gpio/;echo 55 > export;echo out > gpio55/direction;echo 1 > gpio55/value");

    switch (*argv[1]) {
        case '0':
            std::cout << "i am 1";
            SAMPLE_MEIDA_OPEN();
            break;
        default:
            SAMPLE_AI_Usage(argv[0]);
            break;
    }
    sdk_exit();
    SAMPLE_PRT("\nsdk exit success\n");
    return s32Ret;



}

