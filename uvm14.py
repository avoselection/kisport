#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
УВМ-14: ассемблер (JSON-ASM) → машинный код (112-бит/14 байт), интерпретатор,
CLI и простое Tk GUI. Соответствует заданиям (Этапы 1–6) и примерам со
спецификацией, где поля инструкций:

Общее: размер команды всегда 14 байт (112 бит). Поля нумеруются в битах.
A — биты 0–6 (7 бит) — код операции.
B — биты 7–35 (29 бит) — адрес/назначение (в зависимости от операции).

Дальнейшие поля зависят от кода A (как в методичке):
- A = 3   (Загрузка константы)        : C = биты 36–65 (30 бит) — константа.
- A = 55  (Чтение значения из памяти) : C = биты 36–64 (29 бит) — адрес-источник.
- A = 56  (Запись значения в память)  : C = биты 36–49 (14 бит) — смещение,
                                        D = биты 50–78 (29 бит) — базовый адрес.
- A = 78  (Бинарная операция: pow())  : C = биты 36–49 (14 бит) — смещение,
                                        D = биты 50–78 (29 бит) — адрес операнда 1 (база результата),
                                        E = биты 79–107 (29 бит) — адрес операнда 2.

Принята модель данных: память данных — массив целых чисел (одна ячейка = одно
целое без ограничения размера в рамках Python). Память команд — отдельный
байтовый массив.

JSON-ASM (человекочитаемый язык): массив объектов. Допускаются две формы:
  1) Низкоуровневая: {"A": 3, "B": 625, "C": 1013}
  2) Удобная с именами: {"op": "LDC", "B": 625, "C": 1013}
Поддержанные op: LDC (A=3), LD (A=55), ST (A=56), POW (A=78).

CLI:
  Ассемблер:
    python uvm14.py assemble program.json out.bin [--test]
      --test  → печать IR (список полей/значений) и байт-код.

  Интерпретатор:
    python uvm14.py run out.bin --dump-csv mem.csv --range 0:127 [--test]
      --dump-csv  → путь для CSV-дампа памяти данных после выполнения.
      --range A:B → диапазон адресов (включительно) для дампа.
      --test      → печать числа исполненных инструкций.

Tk GUI:
  python uvm14.py gui
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# =============================
# Битовые помощники
# =============================

def set_bits(value: int, start: int, end: int, field: int) -> int:
    """Вставить field в value в биты [start..end] (включительно), little-endian по весам."""
    width = end - start + 1
    mask = ((1 << width) - 1) << start
    return (value & ~mask) | ((field & ((1 << width) - 1)) << start)


def get_bits(value: int, start: int, end: int) -> int:
    width = end - start + 1
    return (value >> start) & ((1 << width) - 1)

# =============================
# Промежуточное представление (IR)
# =============================

OP_BY_NAME = {
    'LDC': 3,
    'LD': 55,
    'ST': 56,
    'POW': 78,
}

NAME_BY_OP = {v: k for k, v in OP_BY_NAME.items()}

@dataclass
class IR:
    A: int
    B: int = 0
    C: Optional[int] = None
    D: Optional[int] = None
    E: Optional[int] = None
    # сохраняем исходный индекс/смещение при необходимости (для отладки)
    off: Optional[int] = None

    def fields(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"A": self.A, "B": self.B}
        if self.C is not None: d["C"] = self.C
        if self.D is not None: d["D"] = self.D
        if self.E is not None: d["E"] = self.E
        return d

# =============================
# Ассемблер JSON → IR → 14-байт
# =============================

class Assembler14:
    def parse_json(self, data: List[Dict[str, Any]]) -> List[IR]:
        ir: List[IR] = []
        for i, item in enumerate(data):
            if 'A' in item:
                A = int(item['A'])
            else:
                op = str(item.get('op', '')).upper()
                if op not in OP_BY_NAME:
                    raise ValueError(f"Неизвестный op: {op}")
                A = OP_BY_NAME[op]
            B = int(item.get('B', 0))
            C = item.get('C')
            D = item.get('D')
            E = item.get('E')
            ir.append(IR(A=A, B=B, C=(None if C is None else int(C)),
                         D=(None if D is None else int(D)),
                         E=(None if E is None else int(E)), off=i*14))
        return ir

    def encode_one(self, ir: IR) -> bytes:
        A = ir.A & 0x7F
        B = ir.B & ((1 << 29) - 1)
        v = 0
        v = set_bits(v, 0, 6, A)
        v = set_bits(v, 7, 35, B)
        if ir.A == 3:  # LDC
            C = (ir.C or 0) & ((1 << 30) - 1)
            v = set_bits(v, 36, 65, C)
        elif ir.A == 55:  # LD
            C = (ir.C or 0) & ((1 << 29) - 1)
            v = set_bits(v, 36, 64, C)
        elif ir.A == 56:  # ST
            C = (ir.C or 0) & ((1 << 14) - 1)
            D = (ir.D or 0) & ((1 << 29) - 1)
            v = set_bits(v, 36, 49, C)
            v = set_bits(v, 50, 78, D)
        elif ir.A == 78:  # POW
            C = (ir.C or 0) & ((1 << 14) - 1)
            D = (ir.D or 0) & ((1 << 29) - 1)
            E = (ir.E or 0) & ((1 << 29) - 1)
            v = set_bits(v, 36, 49, C)
            v = set_bits(v, 50, 78, D)
            v = set_bits(v, 79, 107, E)
        else:
            # прочие коды не используются в данной спецификации
            pass
        return v.to_bytes(14, 'little')

    def encode(self, ir_list: List[IR]) -> bytes:
        out = bytearray()
        for instr in ir_list:
            out += self.encode_one(instr)
        return bytes(out)

    def decode_one(self, data14: bytes) -> IR:
        if len(data14) != 14:
            raise ValueError('Инструкция должна быть 14 байт')
        v = int.from_bytes(data14, 'little')
        A = get_bits(v, 0, 6)
        B = get_bits(v, 7, 35)
        C = D = E = None
        if A == 3:
            C = get_bits(v, 36, 65)
        elif A == 55:
            C = get_bits(v, 36, 64)
        elif A == 56:
            C = get_bits(v, 36, 49)
            D = get_bits(v, 50, 78)
        elif A == 78:
            C = get_bits(v, 36, 49)
            D = get_bits(v, 50, 78)
            E = get_bits(v, 79, 107)
        return IR(A=A, B=B, C=C, D=D, E=E)

# =============================
# Интерпретатор
# =============================

class UVM14:
    """Память команд (bytes) отделена от памяти данных (int-ячейки)."""
    def __init__(self, data_mem_size: int = 1<<16):
        self.code: bytes = b''
        self.pc: int = 0  # байтовый индекс в code
        self.data: List[int] = [0] * data_mem_size
        self.halted: bool = False
        self.asm = Assembler14()

    def load_code(self, code: bytes, *, reset_data: bool = False):
        self.code = bytes(code)
        self.pc = 0
        self.halted = False
        if reset_data:
            self.data = [0] * len(self.data)

    def fetch14(self) -> Optional[bytes]:
        if self.pc + 14 > len(self.code):
            return None
        b = self.code[self.pc:self.pc+14]
        self.pc += 14
        return b

    def step(self) -> bool:
        if self.halted:
            return False
        instr_bytes = self.fetch14()
        if not instr_bytes:
            self.halted = True
            return False
        ir = self.asm.decode_one(instr_bytes)
        A,B,C,D,E = ir.A, ir.B, ir.C, ir.D, ir.E
        # Выполнение согласно спецификации
        if A == 3:      # LDC: mem[B] = C
            self._ensure_addr(B)
            self.data[B] = int(C or 0)
        elif A == 55:   # LD:  mem[B] = mem[C]
            self._ensure_addr(B); self._ensure_addr(C)
            self.data[B] = self.data[int(C)]
        elif A == 56:   # ST:  mem[D + C] = mem[B]
            addr = int(D or 0) + int(C or 0)
            self._ensure_addr(addr); self._ensure_addr(B)
            self.data[addr] = self.data[int(B)]
        elif A == 78:   # POW: mem[D + C] = pow(mem[E], mem[B])
            addr = int(D or 0) + int(C or 0)
            self._ensure_addr(addr); self._ensure_addr(E); self._ensure_addr(B)
            base = self.data[int(E)]
            exp  = self.data[int(B)]
            self.data[addr] = pow(base, exp)
        else:
            # Неподдержанная операция — останавливаемся
            self.halted = True
            return False
        return True

    def run(self, max_steps: int = 10_000_000) -> int:
        steps = 0
        while not self.halted and steps < max_steps:
            if not self.step():
                break
            steps += 1
        return steps

    def _ensure_addr(self, a: Optional[int]):
        if a is None:
            raise ValueError('Ожидался адрес, получено None')
        if a < 0 or a >= len(self.data):
            raise IndexError(f'Адрес вне диапазона данных: {a}')

# =============================
# Утилиты печати и дампов
# =============================

def pretty_ir(ir_list: List[IR]) -> str:
    lines: List[str] = []
    for i, ir in enumerate(ir_list):
        fields = ir.fields()
        pairs = ', '.join(f"{k}={v}" for k, v in fields.items())
        lines.append(f"{i:03}: {pairs}")
    return '\n'.join(lines)


def hexdump(code: bytes) -> str:
    return ' '.join(f"0x{b:02X}" for b in code)


def dump_csv(data_mem: List[int], start: int, end: int, path: str):
    if start > end:
        start, end = end, start
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['addr', 'value'])
        for a in range(start, end + 1):
            w.writerow([a, data_mem[a]])

# =============================
# CLI
# =============================

def cli(argv=None):
    p = argparse.ArgumentParser(prog='uvm14', description='УВМ-14: ассемблер и интерпретатор (14-байтовые команды)')
    sub = p.add_subparsers(dest='cmd', required=True)

    pa = sub.add_parser('assemble', help='Ассемблировать JSON → bin (14-байтовые инструкции)')
    pa.add_argument('src', help='program.json')
    pa.add_argument('out', help='out.bin')
    pa.add_argument('--test', action='store_true', help='печать IR и байт-кода')

    pr = sub.add_parser('run', help='Исполнить бинарник')
    pr.add_argument('bin', help='in.bin')
    pr.add_argument('--dump-csv', required=True, help='путь для CSV дампа памяти данных')
    pr.add_argument('--range', dest='rng', default='0:127', help='диапазон адресов для дампа, формат A:B (включительно)')
    pr.add_argument('--test', action='store_true', help='печать числа инструкций')
    pr.add_argument('--mem-size', type=int, default=1<<16, help='размер памяти данных (ячейки)')

    pg = sub.add_parser('gui', help='GUI (Tk)')

    args = p.parse_args(argv)

    if args.cmd == 'assemble':
        data = json.load(open(args.src, 'r', encoding='utf-8'))
        asm = Assembler14()
        ir = asm.parse_json(data)
        code = asm.encode(ir)
        open(args.out, 'wb').write(code)
        print(f'assembled_instructions={len(ir)}')
        if args.test:
            print('IR:')
            print(pretty_ir(ir))
            print('Bytes:')
            print(hexdump(code))
    elif args.cmd == 'run':
        code = open(args.bin, 'rb').read()
        vm = UVM14(data_mem_size=args.mem_size)
        vm.load_code(code)
        steps = vm.run()
        if args.test:
            print(f'executed_instructions={steps}')
            print(f'code_bytes={len(code)}')
        a, b = [int(x) for x in args.rng.split(':')]
        dump_csv(vm.data, a, b, args.dump_csv)
        print(f'CSV dump → {args.dump_csv} [{a}:{b}]')
    elif args.cmd == 'gui':
        run_gui()

# =============================
# Простое GUI на Tkinter
# =============================

def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title('УВМ-14 — Ассемблер и Интерпретатор')
    root.geometry('1000x700')

    frm = ttk.Frame(root, padding=8)
    frm.pack(fill='both', expand=True)

    txt = tk.Text(frm, wrap='none')
    txt.pack(side='left', fill='both', expand=True)
    txt.insert('1.0', '''[
  {"op":"LDC","B":625,"C":1013},
  {"op":"LD","B":10,"C":625},
  {"op":"ST","B":10,"C":5,"D":200},
  {"op":"POW","B":2,"C":0,"D":300,"E":301}
]''')

    vs = ttk.Scrollbar(frm, orient='vertical', command=txt.yview)
    vs.pack(side='left', fill='y')
    txt.configure(yscrollcommand=vs.set)

    right = ttk.Frame(frm, padding=(8,0))
    right.pack(side='left', fill='y')

    out = tk.Text(right, height=20, width=40)
    out.pack(fill='both', expand=True)

    def assemble_and_run():
        try:
            data = json.loads(txt.get('1.0', 'end'))
            asm = Assembler14()
            ir = asm.parse_json(data)
            code = asm.encode(ir)
            vm = UVM14()
            vm.load_code(code)
            steps = vm.run()
            out.delete('1.0', 'end')
            out.insert('end', 'IR:\n'+pretty_ir(ir)+'\n\n')
            out.insert('end', 'Bytes:\n'+hexdump(code)+'\n\n')
            out.insert('end', f'Executed: {steps} steps\n')
            # показать кусок памяти 0..40
            snippet = '\n'.join(f"{i}: {vm.data[i]}" for i in range(0, 41))
            out.insert('end', 'Memory[0..40]:\n'+snippet+'\n')
        except Exception as e:
            messagebox.showerror('Ошибка', str(e))

    ttk.Button(right, text='Ассемблировать и выполнить', command=assemble_and_run).pack(fill='x')

    def save_dump_as():
        try:
            path = filedialog.asksaveasfilename(defaultextension='.csv')
            if not path:
                return
            data = json.loads(txt.get('1.0', 'end'))
            asm = Assembler14()
            code = asm.encode(asm.parse_json(data))
            vm = UVM14()
            vm.load_code(code)
            vm.run()
            dump_csv(vm.data, 0, 127, path)
            out.insert('end', f'Dump saved: {path}\n')
        except Exception as e:
            messagebox.showerror('Ошибка', str(e))

    ttk.Button(right, text='Сохранить CSV дамп 0..127', command=save_dump_as).pack(fill='x')

    root.mainloop()

# =============================
# Точка входа
# =============================

if __name__ == '__main__':
    cli()
