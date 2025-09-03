#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
УВМ: ассемблер (JSON-ASM) → машинный код, интерпретатор, CLI, Tk GUI.
Этапы 1–5 покрыты. Веб-GUI — в web/index.html (PyScript/WASM).
"""
from __future__ import annotations
import argparse
import json
import struct
import sys
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union

# =============================
# ISA и кодировки
# =============================
REGS = {f"R{i}": i for i in range(8)}

OPCODES = {
    'LDC': 0x01,
    'LD':  0x02,
    'ST':  0x03,
    'BSWAP':0x04,
    'ADD': 0x05,
    'SUB': 0x06,
    'JMP': 0x07,
    'JNZ': 0x08,
    'MOV': 0x09,
    'LABEL': 0x00,  # псевдоинструкция
}

@dataclass
class IRInstr:
    op: str
    dst: Optional[int] = None
    src: Optional[int] = None
    reg: Optional[int] = None
    imm: Optional[int] = None
    addr: Optional[Union[int,str]] = None
    label: Optional[str] = None
    pc: Optional[int] = None  # адрес в коде (байты)

    def fields(self) -> Dict[str, Any]:
        return {k:v for k,v in self.__dict__.items() if v is not None}

# =============================
# Ассемблер JSON → IR → байты
# =============================
class Assembler:
    def __init__(self):
        self.labels: Dict[str,int] = {}
        self.ir: List[IRInstr] = []

    def parse_json(self, data: List[Dict[str,Any]]) -> List[IRInstr]:
        ir: List[IRInstr] = []
        pc = 0
        # первая проходка: собрать метки и черновую IR
        for item in data:
            op = item.get('op')
            if not op:
                raise ValueError('Каждый объект должен иметь поле op')
            op = op.upper()
            if op == 'LABEL':
                name = item.get('name')
                if not name:
                    raise ValueError('LABEL требует поле name')
                if name in self.labels:
                    raise ValueError(f'Повторная метка {name}')
                self.labels[name] = pc
                ir.append(IRInstr(op='LABEL', label=name, pc=pc))
                continue
            ii = IRInstr(op=op, pc=pc)
            # регистры могут быть строками
            if 'dst' in item: ii.dst = self._reg(item['dst'])
            if 'src' in item: ii.src = self._reg(item['src'])
            if 'reg' in item: ii.reg = self._reg(item['reg'])
            if 'imm' in item: ii.imm = self._u32(item['imm'])
            if 'addr' in item: ii.addr = item['addr']
            if 'target' in item: ii.addr = item['target']
            ir.append(ii)
            pc += self._size_of(ii)
        self.ir = ir
        return ir

    def resolve_and_encode(self) -> bytes:
        # вторую проходку резолвим адреса/метки и кодируем
        code = bytearray()
        pc = 0
        for ii in self.ir:
            if ii.op == 'LABEL':
                continue
            opcode = OPCODES.get(ii.op)
            if opcode is None or opcode == 0x00:
                raise ValueError(f'Неизвестная оп: {ii.op}')
            regbyte = ((ii.dst if ii.dst is not None else 0) & 0xF) << 4
            low = (ii.src if ii.src is not None else (ii.reg if ii.reg is not None else 0)) & 0xF
            regbyte |= low
            code.append(opcode)
            code.append(regbyte)
            need_imm = ii.op in ('LDC','LD','ST','JMP','JNZ')
            if need_imm:
                imm = self._encode_addr_or_imm(ii)
                code += struct.pack('<I', imm)
            pc += 2 + (4 if need_imm else 0)
        return bytes(code)

    # ===== helpers =====
    def _reg(self, val: Union[str,int]) -> int:
        if isinstance(val,int):
            if 0 <= val < 8: return val
            raise ValueError('Номер регистра вне диапазона 0..7')
        if isinstance(val,str):
            val = val.upper()
            if val in REGS: return REGS[val]
            raise ValueError(f'Неизвестный регистр {val}')
        raise TypeError('Ожидался регистр (строка или int)')

    def _u32(self, x: int) -> int:
        if not isinstance(x,int):
            raise TypeError('imm должен быть целым числом')
        return x & 0xFFFFFFFF

    def _size_of(self, ii: IRInstr) -> int:
        if ii.op == 'LABEL': return 0
        return 2 + (4 if ii.op in ('LDC','LD','ST','JMP','JNZ') else 0)

    def _encode_addr_or_imm(self, ii: IRInstr) -> int:
        # Поле addr/target/imm может быть: число, имя метки, имя регистра (непрямая адресация)
        if ii.op == 'LDC':
            return self._u32(ii.imm if ii.imm is not None else 0)
        # LD/ST/JMP/JNZ
        operand = ii.addr if ii.addr is not None else 0
        if isinstance(operand,int):
            return operand & 0xFFFFFFFF
        if isinstance(operand,str):
            operand = operand.upper()
            if operand in self.labels:
                return self.labels[operand] & 0xFFFFFFFF
            if operand in REGS:
                # специальный маркер: высокие 16 бит = 0xFFFF → интерпретатор понимает как непрямая адресация из регистра
                return ((0xFFFF << 16) | (REGS[operand] & 0xFFFF)) & 0xFFFFFFFF
            raise ValueError(f'Неизвестная метка/рег {operand}')
        raise TypeError('addr/target должен быть числом или строкой')

# =============================
# Интерпретатор бинарника
# =============================
class UVM:
    def __init__(self, mem_size: int = 64*1024):
        self.mem = bytearray(mem_size)
        self.reg = [0]*8
        self.pc = 0
        self.halted = False

    def load_code(self, code: bytes, at: int = 0):
        self.mem[at:at+len(code)] = code
        self.pc = at

    def _read_u32(self, addr: int) -> int:
        return struct.unpack_from('<I', self.mem, addr)[0]

    def _write_u32(self, addr: int, val: int):
        struct.pack_into('<I', self.mem, addr, val & 0xFFFFFFFF)

    def step(self) -> bool:
        if self.halted: return False
        opcode = self.mem[self.pc]
        regbyte = self.mem[self.pc+1]
        dst = (regbyte >> 4) & 0xF
        low = regbyte & 0xF
        self.pc += 2

        def fetch_imm():
            nonlocal self
            imm = self._read_u32(self.pc)
            self.pc += 4
            return imm

        if opcode == 0x01:  # LDC
            imm = fetch_imm()
            self.reg[dst] = imm
        elif opcode == 0x02:  # LD
            op = fetch_imm()
            addr = self._resolve_addr(op)
            self.reg[dst] = self._read_u32(addr)
        elif opcode == 0x03:  # ST
            op = fetch_imm()
            addr = self._resolve_addr(op)
            src = low & 0x7
            self._write_u32(addr, self.reg[src])
        elif opcode == 0x04:  # BSWAP
            # перестановка байт 32-битного регистра
            x = self.reg[dst] & 0xFFFFFFFF
            b0 = (x >> 0) & 0xFF
            b1 = (x >> 8) & 0xFF
            b2 = (x >> 16) & 0xFF
            b3 = (x >> 24) & 0xFF
            self.reg[dst] = (b0<<24) | (b1<<16) | (b2<<8) | b3
        elif opcode == 0x05:  # ADD
            src = low & 0x7
            self.reg[dst] = (self.reg[dst] + self.reg[src]) & 0xFFFFFFFF
        elif opcode == 0x06:  # SUB
            src = low & 0x7
            self.reg[dst] = (self.reg[dst] - self.reg[src]) & 0xFFFFFFFF
        elif opcode == 0x07:  # JMP
            target = fetch_imm()
            self.pc = self._resolve_addr(target)
        elif opcode == 0x08:  # JNZ
            target = fetch_imm()
            reg = low & 0x7
            if self.reg[reg] != 0:
                self.pc = self._resolve_addr(target)
        elif opcode == 0x09:  # MOV
            src = low & 0x7
            self.reg[dst] = self.reg[src]
        else:
            # простейший halt по 0x00 или неизвестному коду
            self.halted = True
            return False
        return True

    def run(self, max_steps: int = 10_000_000):
        steps = 0
        while not self.halted and steps < max_steps:
            if not self.step():
                break
            steps += 1
        return steps

    def _resolve_addr(self, op: int) -> int:
        # Непрямая адресация: если high16 == 0xFFFF, low16 — номер регистра
        if (op >> 16) == 0xFFFF:
            r = op & 0xFFFF
            return self.reg[r & 0x7] & 0xFFFFFFFF
        return op & 0xFFFFFFFF

# =============================
# Тестовый вывод (этапы 1–2)
# =============================

def pretty_ir(ir: List[IRInstr]) -> str:
    lines = []
    for i, ii in enumerate(ir):
        if ii.op == 'LABEL':
            lines.append(f"{i:03}: LABEL name={ii.label} pc={ii.pc}")
        else:
            fields = ii.fields()
            ftxt = ', '.join(f"{k}={v}" for k,v in fields.items())
            lines.append(f"{i:03}: {ftxt}")
    return '\n'.join(lines)


def hexdump(data: bytes) -> str:
    return ' '.join(f"{b:02X}" for b in data)

# =============================
# XML дамп памяти
# =============================

def dump_memory_xml(mem: bytes, start: int, end: int, path: str):
    root = ET.Element('memory', start=str(start), end=str(end))
    for addr in range(start, end+1):
        cell = ET.SubElement(root, 'cell', addr=str(addr))
        cell.text = f"{mem[addr]:02X}"
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)

# =============================
# CLI
# =============================

def cli(argv=None):
    p = argparse.ArgumentParser(prog='uvm')
    sub = p.add_subparsers(dest='cmd', required=True)

    pa = sub.add_parser('assemble', help='Ассемблировать JSON → bin')
    pa.add_argument('src', help='program.json')
    pa.add_argument('out', help='out.bin')
    pa.add_argument('--test', action='store_true', help='печать IR в стиле полей/значений')
    pa.add_argument('--bytes', action='store_true', help='печать байт-кода (тест из спецификации)')

    pr = sub.add_parser('run', help='Исполнить бинарник')
    pr.add_argument('bin', help='in.bin')
    pr.add_argument('--dump-xml', help='путь для XML дампа памяти')
    pr.add_argument('--range', dest='rng', default='0:127', help='диапазон адресов для дампа, формат A:B (включительно)')
    pr.add_argument('--test', action='store_true', help='печать результата ассемблирования в байтах (если доступно) и число инструкций')

    pg = sub.add_parser('gui', help='GUI (Tk)')

    pb = sub.add_parser('build', help='Сборка дистрибутивов')
    pb.add_argument('--all', action='store_true')

    args = p.parse_args(argv)

    if args.cmd == 'assemble':
        data = json.load(open(args.src,'r',encoding='utf-8'))
        asm = Assembler()
        ir = asm.parse_json(data)
        code = asm.resolve_and_encode()
        open(args.out,'wb').write(code)
        if args.test:
            print(pretty_ir(ir))
        if args.bytes:
            print(hexdump(code))
    elif args.cmd == 'run':
        code = open(args.bin,'rb').read()
        vm = UVM()
        vm.load_code(code)
        steps = vm.run()
        if args.test:
            print(f"executed_instructions={steps}")
            print(f"code_bytes={len(code)}")
        if args.dump_xml:
            a,b = [int(x) for x in args.rng.split(':')]
            dump_memory_xml(vm.mem, a, b, args.dump_xml)
            print(f"XML dump → {args.dump_xml}")
    elif args.cmd == 'gui':
        run_gui()
    elif args.cmd == 'build':
        if args.all:
            print('Запуск PyInstaller…')
            import subprocess
            subprocess.check_call([sys.executable,'-m','pip','install','pyinstaller'])
            subprocess.check_call(['pyinstaller','-F','uvm.py'])
            print('Готово: dist/uvm')
        else:
            print('Укажите --all')

# =============================
# Простое GUI на Tkinter (кроссплатформенное)
# =============================

def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title('УВМ — Ассемблер и Интерпретатор')
    root.geometry('1000x700')

    frm = ttk.Frame(root, padding=8)
    frm.pack(fill='both', expand=True)

    txt = tk.Text(frm, wrap='none')
    txt.pack(side='left', fill='both', expand=True)
    txt.insert('1.0','''[
  {"op":"LDC","dst":"R1","imm":0x11223344},
  {"op":"BSWAP","dst":"R1"},
  {"op":"ST","src":"R1","addr":0}
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
            data = json.loads(txt.get('1.0','end'))
            asm = Assembler()
            ir = asm.parse_json(data)
            code = asm.resolve_and_encode()
            vm = UVM()
            vm.load_code(code)
            steps = vm.run()
            a,b = 0, 127
            tmpxml = 'dump.xml'
            dump_memory_xml(vm.mem, a, b, tmpxml)
            out.delete('1.0','end')
            out.insert('end', 'IR:\n'+pretty_ir(ir)+'\n\n')
            out.insert('end', 'Bytes:\n'+hexdump(code)+'\n\n')
            out.insert('end', f'Executed: {steps} steps\n')
            out.insert('end', f'Dump saved: {tmpxml}\n')
        except Exception as e:
            messagebox.showerror('Ошибка', str(e))

    ttk.Button(right, text='Ассемблировать и выполнить', command=assemble_and_run).pack(fill='x')
    ttk.Button(right, text='Сохранить XML дамп как…', command=lambda: save_dump_as()).pack(fill='x')

    def save_dump_as():
        path = filedialog.asksaveasfilename(defaultextension='.xml')
        if not path:
            return
        vm = UVM()
        dump_memory_xml(vm.mem, 0, 127, path)
        messagebox.showinfo('Готово', f'Сохранено: {path}')

    root.mainloop()

# =============================
# Точка входа
# =============================
if __name__ == '__main__':
    cli()