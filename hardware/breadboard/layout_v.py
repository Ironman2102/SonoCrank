"""Spread-out vertical layout for a 62-row GL-12 board, ESP32 OFF the board.
Columns left to right: L+ (3.3 V)  L- (GND) | a b c d e | channel | f g h i j | R- (GND)  R+ (5 V).
Rows 1..62 top to bottom. Spacing rules: flat resistors span 4 rows, parts that overlap in rows sit
at least 2 columns apart, every part going to a rail gets its own row, long links are flat jumper wires."""
U1 = {("e",24):1, ("e",25):2, ("e",26):3, ("e",27):4, ("f",24):8, ("f",25):7, ("f",26):6, ("f",27):5}   # notch toward row 1

PARTS = [
 # step 2: VREF - one rail part per row, rows 3/6/9/12 joined by short wires in column e
 ("R_TOP","res","15 kΩ",2,[("1",("L+",3)),("2",("a",3))],"Stands upright: one leg in the 3.3 V rail, one in a3"),
 ("R_BOT","res","15 kΩ",2,[("1",("a",6)),("2",("L-",6))],"Stands upright: a6 to the GND rail"),
 ("C_REF","ecap","10 µF",2,[("+",("a",9)),("-",("L-",9))],"Long leg (+) in a9, short leg in the GND rail"),
 ("C_REF2","cap","0.1 µF",2,[("1",("a",12)),("2",("L-",12))],"a12 to the GND rail (marked 104)"),
 # step 3: op-amp supply - pin 8 row 24, extended up to rows 21 and 18 for the capacitors
 ("R_SUP","res","47 Ω",3,[("1",("R+",24)),("2",("j",24))],"Stands upright: 5 V rail to j24 (pin 8 row)"),
 ("C_SUP","ecap","100 µF",3,[("+",("j",21)),("-",("R-",21))],"Long leg (+) in j21, short leg in the GND rail"),
 ("C_SUP2","cap","0.1 µF",3,[("1",("j",18)),("2",("R-",18))],"j18 to the GND rail (marked 104)"),
 # step 4: amplifier (left half)
 ("R_BIAS","res","560 kΩ",4,[("1",("c",9)),("2",("c",13))],"Lies flat in column c, rows 9 to 13"),
 ("PZ","piezo","Piezo disc",4,[("black",("b",9)),("red",("b",13))],"Black wire → b9, red wire → b13 (enter from the left)"),
 ("R_IN","res","12 kΩ",4,[("1",("d",13)),("2",("d",17))],"Lies flat in column d, rows 13 to 17"),
 ("C_G","ecap","10 µF",4,[("+",("d",20)),("-",("d",19))],"Stands up: long leg (+) d20, short leg d19"),
 ("R_G","res","12 kΩ",4,[("1",("c",20)),("2",("c",25))],"Lies flat in column c, rows 20 to 25"),
 ("R_F","res","56 kΩ",4,[("1",("a",24)),("2",("a",25))],"Stands upright in a24 and a25 (swap this to change gain)"),
 # step 5: filter (right half, below the chip)
 ("R_1","res","8.2 kΩ",5,[("1",("e",33)),("2",("g",33))],"e33 → g33, the only part crossing the channel"),
 ("R_2","res","8.2 kΩ",5,[("2",("f",29)),("1",("f",33))],"Lies flat in column f, rows 29 to 33"),
 ("C_2","cap","3.3 nF",5,[("1",("j",29)),("2",("R-",29))],"j29 to the GND rail (marked 332)"),
 ("C_1","cap","5.6 nF",5,[("2",("h",31)),("1",("h",33))],"Stands in h31 and h33 (marked 562)"),
 ("R_OUT","res","1 kΩ",5,[("1",("j",31)),("2",("j",35))],"Lies flat in column j, rows 31 to 35"),
 # step 6: outputs, spaced out, top to bottom in ESP32 pin order (D25, D26, D27, D14)
 ("R_B","res","1 kΩ",6,[("1",("i",40)),("2",("i",44))],"Lies flat in column i, rows 40 to 44"),
 ("Q1","npn","BC548",6,[("C",("g",43)),("B",("g",44)),("E",("g",45))],"Flat face toward the channel: C g43, B g44, E g45"),
 ("BZ","buzzer","B-27 buzzer",6,[("-",("j",43)),("+",("R+",41))],"Black wire → j43, red wire → 5 V rail"),
 ("LED_R","led","Red LED",6,[("A",("h",49)),("K",("h",50))],"Long leg (+) h49, short leg h50"),
 ("R_LR","res","270 Ω",6,[("1",("j",50)),("2",("R-",51))],"Stands upright: j50 to the GND rail"),
 ("LED_Y","led","Yellow LED",6,[("A",("h",54)),("K",("h",55))],"Long leg (+) h54, short leg h55"),
 ("R_LY","res","270 Ω",6,[("1",("j",55)),("2",("R-",56))],"Stands upright: j55 to the GND rail"),
 ("BTN","button","Push button",6,[("A",("g",59)),("B",("g",61)),("C",("d",59)),("D",("d",61))],"Across the channel, rows 59 and 61"),
]
JUMPERS = [
 ("W1",1,("ESP","3V3"),("L+",1),"pwr3","ESP32 3V3 → left + rail at row 1 (route over the top end of the board)"),
 ("W2",1,("ESP","GND2"),("L-",2),"gnd","ESP32 GND (3V3 side) → left − rail at row 2"),
 ("W3",1,("ESP","VIN"),("R+",62),"pwr5","ESP32 VIN → right + rail at row 62"),
 ("W4",1,("ESP","GND1"),("R-",61),"gnd","ESP32 GND (VIN side) → right − rail at row 61"),
 ("J5",1,("L-",0),("R-",0),"gnd","Joins both GND rails across the top end"),
 ("J6",2,("e",3),("d",6),"ref","VREF link e3 → d6"),
 ("J7",2,("e",6),("d",9),"ref","VREF link e6 → d9"),
 ("J8",2,("e",9),("d",12),"ref","VREF link e9 → d12"),
 ("J9",4,("e",12),("e",19),"ref","VREF down to row 19 (lies flat along column e)"),
 ("J10",3,("a",27),("L-",27),"gnd","a27 (pin 4) → GND rail"),
 ("J11",3,("h",24),("h",21),"sig","Pin 8 row up to row 21"),
 ("J12",3,("g",21),("g",18),"sig","g21 up to g18"),
 ("J13",4,("b",17),("b",26),"sig","b17 → b26 (to pin 3), lies flat along column b"),
 ("J14",5,("d",24),("d",33),"sig","d24 (pin 1) → d33, lies flat along column d"),
 ("J15",5,("g",25),("g",26),"sig","g25 (pin 7) → g26 (pin 6)"),
 ("J16",5,("h",25),("g",31),"sig","h25 (pin 7) → g31"),
 ("J17",5,("i",27),("i",29),"sig","i27 (pin 5) → i29"),
 ("W5",5,("i",35),("ESP","D34"),"adc","i35 → ESP32 D34 (short, kept apart from the other wires)"),
 ("W6",6,("ESP","D25"),("j",40),"io","ESP32 D25 → j40 (buzzer)"),
 ("J18",6,("j",45),("R-",45),"gnd","j45 (emitter) → GND rail"),
 ("W7",6,("ESP","D26"),("j",49),"io","ESP32 D26 → j49 (red LED)"),
 ("W8",6,("ESP","D27"),("j",54),"io","ESP32 D27 → j54 (yellow LED)"),
 ("W9",6,("ESP","D14"),("j",59),"io","ESP32 D14 → j59 (button)"),
 ("J19",6,("a",61),("L-",61),"gnd","a61 → GND rail (button)"),
]
