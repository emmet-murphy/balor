/*
*   Copyright (c) 2007 Ilya O. Levin, http://www.literatecode.com
*
*   Permission to use, copy, modify, and distribute this software for any
*   purpose with or without fee is hereby granted, provided that the above
*   copyright notice and this permission notice appear in all copies.
*
*   THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
*   WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
*   MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
*   ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
*   WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
*   ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
*   OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/


#include <cstdint>

uint8_t rj_xtime(uint8_t x) { return (x & 0x80) ? ((x << 1) ^ 0x1b) : (x << 1); } /* rj_xtime */

void aes_mixColumns(uint8_t *buf) {
    register uint8_t i, a, b, c, d, e;

    mix:for (i = 0; i < 16; i += 4) {
        a = buf[i];
        b = buf[i + 1];
        c = buf[i + 2];
        d = buf[i + 3];
        e = a ^ b ^ c ^ d;
        buf[i] ^= e ^ rj_xtime(a ^ b);
        buf[i + 1] ^= e ^ rj_xtime(b ^ c);
        buf[i + 2] ^= e ^ rj_xtime(c ^ d);
        buf[i + 3] ^= e ^ rj_xtime(d ^ a);
    }
} /* aes_mixColumns */
