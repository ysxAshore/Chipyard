// See LICENSE.Sifive for license details.
#ifndef _SDBOOT_KPRINTF_H
#define _SDBOOT_KPRINTF_H

#include <platform.h>
#include <stdint.h>

#define REG32(p, i)	((p)[(i) >> 2])

#ifndef UART_CTRL_ADDR
  #ifndef UART_NUM
    #define UART_NUM 0
  #endif

  #define _CONCAT3(A, B, C) A ## B ## C
  #define _UART_CTRL_ADDR(UART_NUM) _CONCAT3(UART, UART_NUM, _CTRL_ADDR)
  #define UART_CTRL_ADDR _UART_CTRL_ADDR(UART_NUM)
#endif
volatile static uint32_t *uart_base_ptr = (uint32_t *)(UART_BASE);

static void uart_init()
{
  // enable FIFO
  // 14 bytes trigger  Transmitter FIFO Reset. Receiver FIFO Reset
  *(uart_base_ptr + UART_FCR) = 0x00c7u;

  // set 0x0080 to UART.LCR to enable DLL and DLM write
  // configure baud rate
  *(uart_base_ptr + UART_LCR) = 0x0080u;

  // System clock 70 MHz, 115200 baud rate
  // divisor = clk_freq / (16 * Baud)
  *(uart_base_ptr + UART_DLL) = 25*1000*1000u / (16u * 115200u) % 0x100u;
  *(uart_base_ptr + UART_DLM) = 25*1000*1000u / (16u * 115200u) >> 8;

  // 8-bit data, 1-bit stop
  *(uart_base_ptr + UART_LCR) = 0x03u;

  // Enable read IRQ
  // *(uart_base_ptr + UART_IER) = 0x0001u;
  // Disable all Interrupts
  *(uart_base_ptr + UART_IER) = 0x0;
}

static void uart_send(uint8_t data)
{
  // wait until THR empty
  while (! (*(uart_base_ptr + UART_LSR) & 0x40u));
  *(uart_base_ptr + UART_THR) = data;
}

static char uart_receive()
{
  // wait until RBR full
  while (! (*(uart_base_ptr + UART_LSR) & 0x01u));
  return (char) *(uart_base_ptr + UART_RBR);
}


static inline void kputc(char c)
{
    uart_send(c);
}

static inline void kgetc(char *c)
{
    *c = uart_receive();
}

extern void kputs(const char *);
extern void kprintf(const char *, ...);

#ifdef DEBUG
#define dprintf(s, ...)	kprintf((s), ##__VA_ARGS__)
#define dputs(s)	kputs((s))
#else
#define dprintf(s, ...) do { } while (0)
#define dputs(s)	do { } while (0)
#endif

#endif /* _SDBOOT_KPRINTF_H */
