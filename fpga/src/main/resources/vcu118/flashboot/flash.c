// See LICENSE.Sifive for license details.
#include <stdint.h>

#include <platform.h>

#include "common.h"

#define DEBUG
#include "kprintf.h"

#define SPI_FLASH_BASE_ADDR 0x70000000
#define SPI_FLASH_SIZE 0x8000000 // 128MB
#define SPI_FLASH_PAYLOAD_OFFSET 0x00000400 // 1KB

typedef struct flash_info {//1024 bytes
    uint64_t Magic;         //Magic Number //0xDEADBEEF
    uint64_t ImageSize;     //Image Size
    uint64_t ImageBaseAddr; //Image Base Address,for bootloader payload,copy all of this to DDR

    uint64_t Image_Year;    //the image created year
    uint64_t Image_Month;   //the image created month
    uint64_t Image_Day;     //the image created day
    uint64_t Image_Hour;    //the image created hour
    uint64_t Image_Minute;  //the image created minute

    uint64_t Additional_information_remaining_size; //the remaining size of the additional information
    uint64_t Additional_information[0]; //the additional information
}flash_info_t;




static int flashcopy(void)
{
    //检查第一个KB的内容，里面是我们的magic data
    
    int rc = 0;
    static volatile flash_info_t *flash_info_i = (void *)(SPI_FLASH_BASE_ADDR + 0x00000000);
    static volatile uint64_t *flash_payload = (void *)(SPI_FLASH_BASE_ADDR + SPI_FLASH_PAYLOAD_OFFSET);
    
    if(flash_info_i->Magic != 0xDEADBEEF) {
        kputs("Flash Magic Error!");
        kprintf("Magic: %x\n", flash_info_i->Magic);
        return -1;
    }
    if(flash_info_i->ImageSize > SPI_FLASH_SIZE) {
        kputs("Flash Image Size Error!");
        kprintf("Image Size: %x\n", flash_info_i->ImageSize);
        return -1;
    }

    //将flash内的image信息输出
    kprintf("Flash Image Size: %d\n", flash_info_i->ImageSize);
    kprintf("Next BootLoader Base Address: %x\n", flash_info_i->ImageBaseAddr);
    kprintf("Flash Image Created Time: %d-%d-%d %d:%d\n", flash_info_i->Image_Year, flash_info_i->Image_Month, flash_info_i->Image_Day, flash_info_i->Image_Hour, flash_info_i->Image_Minute);

    //询问是否需要输出剩余的messge信息
    kprintf("Do you want to output the rest of the message[ %d bytes]? (Y/N)\n", flash_info_i->Additional_information_remaining_size);
    char c = 0;
    while (1)
    {
        kgetc(&c);
        if (c == 'Y' || c == 'y')
        {
            kputs("OK, Output the rest of the message!");
            break;
        }
        else if (c == 'N' || c == 'n')
        {
            kputs("OK, Go to copy flash to ddr!");
            break;
        }
    }
    
    //加载flash内的image到DDR
    kputs("Copying flash to DDR...");
    uint64_t *flash_payload_ptr = (uint64_t *)flash_payload;
    uint64_t *ddr_payload_ptr = (uint64_t *)(flash_info_i->ImageBaseAddr);
    uint64_t flash_payload_size = (flash_info_i->ImageSize / 8);
    uint64_t i = 0;
    for (i = 0; i <= flash_payload_size; i++)
    {
        if (i % 256 == 0)//1MB输出一次
        {
            kprintf("Copying 1MB flash to DDR...%d/%d\n", i, flash_payload_size);
        }
        ddr_payload_ptr[i] = flash_payload_ptr[i];
    }
    kprintf("Copying flash to DDR done! %d bytes copied!\n", flash_info_i->ImageSize);
    kprintf("Next BootLoader Base Address: %x\n", flash_info_i->ImageBaseAddr);

	return rc;
}

void truejump2ddrinfo(void)
{
    kputs("[YJP]TrueJump2DDR");
}

int main(void)
{
    //uart初始化
    uart_init();
    

	kputs("INIT");
    kputs("Hello,YJP!");
    kputs("QvQ, Pmod SD Card is Bad...So sad!");
    kputs("OvO, (1)Boot from QSPI Flash in XIP Mode?\n");
    kputs("OvO, (2)Boot from YJP JTAG DDR Booting?\n");
    
    char c = 0;
    while (1)
    {
        kgetc(&c);
        if (c == '1')
        {
            kputs("OK, Booting from QSPI Flash in XIP Mode!");
            flashcopy();
            break;
        }
        else if (c == '2')
        {
            kputs("Wait YJP JTAG DDR Booting!");
            kputs("Do you want to continue? (Y/N)\n");
            while (1)
            {
                kgetc(&c);
                if (c == 'Y' || c == 'y')
                {
                    kputs("OK, YJP JTAG DDR Booting!");
                    break;
                }
                else if (c == 'N' || c == 'n')
                {
                    kputs("OK, Exit!");
                    return 0;
                }
            }
            break;
        }
    }
    

	kputs("BOOT!GOGOGOGOGOGOGOGOGO");
	kputs("Starting the boot process...");

	__asm__ __volatile__ ("fence.i" : : : "memory");

	return 0;
}
