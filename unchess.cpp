// ============================================================================
//  UnChess - Unknown Variable Engine
//  เอนจินหมากรุกที่ "ค้นพบตัวแปรประเมินตำแหน่งใหม่ ๆ ได้เอง"
//  C++17, ไฟล์เดียว ไม่ใช้ไลบรารีภายนอก
//  คอมไพล์:  clang++ -O2 -std=c++17 -pthread unchess.cpp -o unchess
//  (หรือ g++ ก็ได้)
//
//  โหมดใช้งาน:
//    ./unchess                เล่นกับ AI ในเทอร์มินัล
//    ./unchess uci            โหมด UCI (ต่อกับโปรแกรมหมากรุกอื่นได้)
//    ./unchess learn ...      ให้ AI ค้นพบตัวแปรใหม่ด้วยตัวเอง (self-play)
//    ./unchess test           ทดสอบความถูกต้องของกฎ (perft + fuzz)
//    ./unchess perft N [fen]  นับ node
// ============================================================================
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
using namespace std;

typedef uint64_t U64;
typedef int Move;  // from(6) | to(6)<<6 | promo(3)<<12 ; 0 = ไม่มีตา

enum { WHITE = 0, BLACK = 1 };
enum { EMPTY = 0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
static inline int mk(int c, int t) { return c * 8 + t; }
static inline int colorOf(int p) { return p >> 3; }
static inline int typeOf(int p) { return p & 7; }
static inline int popc(U64 x) { return __builtin_popcountll(x); }
static inline int lsb(U64 x) { return __builtin_ctzll(x); }
static inline int poplsb(U64& x) { int s = lsb(x); x &= x - 1; return s; }
static inline U64 bit(int s) { return 1ULL << s; }

static const int MATE = 30000, INF = 32000, MAXPLY = 96;
static const int DIAG[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
static const int ORTH[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
static const int KN[8][2] = {{2, 1}, {2, -1}, {-2, 1}, {-2, -1}, {1, 2}, {1, -2}, {-1, 2}, {-1, -2}};
static const int KG[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
static const char* STARTFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static U64 knightM[64], kingM[64], pawnAtk[2][64], zoneM[64][4], passedM[2][64], fileM[8];
static U64 Z[16][64], ZC[16], ZE[8], ZS;
static int castleMask[64];

static U64 splitmix(U64& s) {
  U64 z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static void initTables() {
  static bool done = false;
  if (done) return;
  done = true;
  for (int sq = 0; sq < 64; sq++) {
    int r = sq >> 3, f = sq & 7;
    knightM[sq] = kingM[sq] = 0;
    pawnAtk[0][sq] = pawnAtk[1][sq] = 0;
    passedM[0][sq] = passedM[1][sq] = 0;
    for (auto& d : KN) {
      int nr = r + d[0], nf = f + d[1];
      if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) knightM[sq] |= bit(nr * 8 + nf);
    }
    for (auto& d : KG) {
      int nr = r + d[0], nf = f + d[1];
      if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) kingM[sq] |= bit(nr * 8 + nf);
    }
    for (int df = -1; df <= 1; df += 2) {
      int nf = f + df;
      if (nf < 0 || nf > 7) continue;
      if (r < 7) pawnAtk[0][sq] |= bit((r + 1) * 8 + nf);
      if (r > 0) pawnAtk[1][sq] |= bit((r - 1) * 8 + nf);
    }
    for (int k = 0; k < 4; k++) {
      zoneM[sq][k] = 0;
      for (int rr = 0; rr < 8; rr++)
        for (int ff = 0; ff < 8; ff++)
          if (max(abs(rr - r), abs(ff - f)) <= k) zoneM[sq][k] |= bit(rr * 8 + ff);
    }
    for (int rr = 0; rr < 8; rr++)
      for (int ff = 0; ff < 8; ff++) {
        if (abs(ff - f) > 1) continue;
        if (rr > r) passedM[0][sq] |= bit(rr * 8 + ff);
        if (rr < r) passedM[1][sq] |= bit(rr * 8 + ff);
      }
  }
  for (int f = 0; f < 8; f++) {
    fileM[f] = 0;
    for (int r = 0; r < 8; r++) fileM[f] |= bit(r * 8 + f);
  }
  U64 seed = 0x1234ABCD5678EF01ULL;
  for (int p = 0; p < 16; p++)
    for (int s = 0; s < 64; s++) Z[p][s] = splitmix(seed);
  for (int i = 0; i < 16; i++) ZC[i] = splitmix(seed);
  for (int i = 0; i < 8; i++) ZE[i] = splitmix(seed);
  ZS = splitmix(seed);
  for (int i = 0; i < 64; i++) castleMask[i] = 15;
  castleMask[4] &= ~3; castleMask[7] &= ~1; castleMask[0] &= ~2;
  castleMask[60] &= ~12; castleMask[63] &= ~4; castleMask[56] &= ~8;
}

// ============================================================================
//  กระดานและกฎหมากรุก
// ============================================================================
struct Pos {
  int8_t b[64];
  int side = 0, castle = 0, ep = -1, half = 0, full = 1;
  U64 key = 0;
  int ksq[2] = {-1, -1};

  Pos() { memset(b, 0, sizeof b); }

  U64 computeKey() const {
    U64 k = 0;
    for (int s = 0; s < 64; s++) if (b[s]) k ^= Z[(int)b[s]][s];
    k ^= ZC[castle];
    if (ep >= 0) k ^= ZE[ep & 7];
    if (side == BLACK) k ^= ZS;
    return k;
  }

  bool setFen(const string& fen) {
    istringstream ss(fen);
    string pl, sd, ca, e;
    int hm = 0, fm = 1;
    if (!(ss >> pl >> sd)) return false;
    if (!(ss >> ca)) ca = "-";
    if (!(ss >> e)) e = "-";
    if (!(ss >> hm)) hm = 0;
    if (!(ss >> fm)) fm = 1;
    Pos n;
    int r = 7, f = 0;
    for (char ch : pl) {
      if (ch == '/') { if (f != 8) return false; r--; f = 0; if (r < 0) return false; }
      else if (ch >= '1' && ch <= '8') { f += ch - '0'; if (f > 8) return false; }
      else {
        int c = isupper((unsigned char)ch) ? WHITE : BLACK, t = 0;
        switch (tolower(ch)) {
          case 'p': t = PAWN; break; case 'n': t = KNIGHT; break; case 'b': t = BISHOP; break;
          case 'r': t = ROOK; break; case 'q': t = QUEEN; break; case 'k': t = KING; break;
          default: return false;
        }
        if (f > 7 || r < 0) return false;
        n.b[r * 8 + f] = (int8_t)mk(c, t);
        f++;
      }
    }
    if (r != 0 || f != 8) return false;
    if (sd == "w") n.side = WHITE; else if (sd == "b") n.side = BLACK; else return false;
    int kc[2] = {0, 0};
    for (int s = 0; s < 64; s++) {
      int p = n.b[s];
      if (!p) continue;
      if (typeOf(p) == KING) { kc[colorOf(p)]++; n.ksq[colorOf(p)] = s; }
      if (typeOf(p) == PAWN && ((s >> 3) == 0 || (s >> 3) == 7)) return false;
    }
    if (kc[0] != 1 || kc[1] != 1) return false;
    n.castle = 0;
    for (char ch : ca) {
      if (ch == 'K' && n.b[4] == mk(0, KING) && n.b[7] == mk(0, ROOK)) n.castle |= 1;
      if (ch == 'Q' && n.b[4] == mk(0, KING) && n.b[0] == mk(0, ROOK)) n.castle |= 2;
      if (ch == 'k' && n.b[60] == mk(1, KING) && n.b[63] == mk(1, ROOK)) n.castle |= 4;
      if (ch == 'q' && n.b[60] == mk(1, KING) && n.b[56] == mk(1, ROOK)) n.castle |= 8;
    }
    n.ep = -1;
    if (e.size() == 2 && e[0] >= 'a' && e[0] <= 'h' && (e[1] == '3' || e[1] == '6'))
      n.ep = (e[1] - '1') * 8 + (e[0] - 'a');
    n.half = max(0, hm); n.full = max(1, fm);
    n.key = n.computeKey();
    // ฝั่งที่ไม่ได้เดินต้องไม่ถูกรุกอยู่ (ตำแหน่งผิดกฎ)
    if (n.attacked(n.ksq[n.side ^ 1], n.side)) return false;
    *this = n;
    return true;
  }

  string fen() const {
    string s;
    for (int r = 7; r >= 0; r--) {
      int e = 0;
      for (int f = 0; f < 8; f++) {
        int p = b[r * 8 + f];
        if (!p) { e++; continue; }
        if (e) { s += char('0' + e); e = 0; }
        char c = " pnbrqk"[typeOf(p)];
        s += colorOf(p) == WHITE ? char(toupper(c)) : c;
      }
      if (e) s += char('0' + e);
      if (r) s += '/';
    }
    s += side == WHITE ? " w " : " b ";
    string c;
    if (castle & 1) c += 'K';
    if (castle & 2) c += 'Q';
    if (castle & 4) c += 'k';
    if (castle & 8) c += 'q';
    s += c.empty() ? "-" : c;
    s += ' ';
    if (ep >= 0) { s += char('a' + (ep & 7)); s += char('1' + (ep >> 3)); } else s += '-';
    s += ' ' + to_string(half) + ' ' + to_string(full);
    return s;
  }

  bool attacked(int sq, int by) const {
    U64 m = pawnAtk[by ^ 1][sq];
    while (m) if (b[poplsb(m)] == mk(by, PAWN)) return true;
    m = knightM[sq];
    while (m) if (b[poplsb(m)] == mk(by, KNIGHT)) return true;
    m = kingM[sq];
    while (m) if (b[poplsb(m)] == mk(by, KING)) return true;
    int r0 = sq >> 3, f0 = sq & 7;
    for (int d = 0; d < 4; d++) {
      int r = r0 + DIAG[d][0], f = f0 + DIAG[d][1];
      while (r >= 0 && r < 8 && f >= 0 && f < 8) {
        int p = b[r * 8 + f];
        if (p) { if (colorOf(p) == by && (typeOf(p) == BISHOP || typeOf(p) == QUEEN)) return true; break; }
        r += DIAG[d][0]; f += DIAG[d][1];
      }
      r = r0 + ORTH[d][0]; f = f0 + ORTH[d][1];
      while (r >= 0 && r < 8 && f >= 0 && f < 8) {
        int p = b[r * 8 + f];
        if (p) { if (colorOf(p) == by && (typeOf(p) == ROOK || typeOf(p) == QUEEN)) return true; break; }
        r += ORTH[d][0]; f += ORTH[d][1];
      }
    }
    return false;
  }
  bool inCheck() const { return attacked(ksq[side], side ^ 1); }

  void make(Move m) {
    int from = m & 63, to = (m >> 6) & 63, pr = (m >> 12) & 7;
    int pc = b[from], us = side, them = us ^ 1, t = typeOf(pc);
    key ^= ZC[castle];
    if (ep >= 0) key ^= ZE[ep & 7];
    int cap = b[to];
    bool isEp = (t == PAWN && (from & 7) != (to & 7) && cap == 0);
    if (cap) key ^= Z[cap][to];
    if (isEp) {
      int cs = us == WHITE ? to - 8 : to + 8;
      key ^= Z[(int)b[cs]][cs];
      b[cs] = 0;
      cap = 1;
    }
    key ^= Z[pc][from];
    b[from] = 0;
    int np = pr ? mk(us, pr) : pc;
    b[to] = (int8_t)np;
    key ^= Z[np][to];
    if (t == KING) {
      ksq[us] = to;
      if (abs(to - from) == 2) {
        int rf, rt;
        if (to > from) { rf = from + 3; rt = from + 1; } else { rf = from - 4; rt = from - 1; }
        int rk = b[rf];
        b[rf] = 0; b[rt] = (int8_t)rk;
        key ^= Z[rk][rf] ^ Z[rk][rt];
      }
    }
    castle &= castleMask[from] & castleMask[to];
    ep = -1;
    if (t == PAWN && abs(to - from) == 16) ep = (from + to) / 2;
    if (t == PAWN || cap) half = 0; else half++;
    if (us == BLACK) full++;
    side = them;
    key ^= ZS;
    key ^= ZC[castle];
    if (ep >= 0) key ^= ZE[ep & 7];
  }

  void makeNull() {
    key ^= ZS;
    if (ep >= 0) { key ^= ZE[ep & 7]; ep = -1; }
    side ^= 1;
    half++;
  }

  // สร้างตาเดินแบบ pseudo-legal (caps = เฉพาะการกิน + เลื่อนขั้นควีน)
  int gen(Move* ml, bool caps) const {
    int n = 0, us = side, them = us ^ 1;
    auto add = [&](int f, int t, int pr = 0) { ml[n++] = f | (t << 6) | (pr << 12); };
    for (int sq = 0; sq < 64; sq++) {
      int pc = b[sq];
      if (!pc || colorOf(pc) != us) continue;
      int t = typeOf(pc), r = sq >> 3, f = sq & 7;
      if (t == PAWN) {
        int dir = us == WHITE ? 1 : -1, promoR = us == WHITE ? 6 : 1, startR = us == WHITE ? 1 : 6;
        int to = sq + dir * 8;
        if (b[to] == 0) {
          if (r == promoR) {
            add(sq, to, QUEEN);
            if (!caps) { add(sq, to, ROOK); add(sq, to, BISHOP); add(sq, to, KNIGHT); }
          } else if (!caps) {
            add(sq, to);
            if (r == startR && b[to + dir * 8] == 0) add(sq, to + dir * 8);
          }
        }
        for (int df = -1; df <= 1; df += 2) {
          int nf = f + df;
          if (nf < 0 || nf > 7) continue;
          int t2 = (r + dir) * 8 + nf;
          if (b[t2] && colorOf(b[t2]) == them) {
            if (r == promoR) { add(sq, t2, QUEEN); add(sq, t2, ROOK); add(sq, t2, BISHOP); add(sq, t2, KNIGHT); }
            else add(sq, t2);
          } else if (ep >= 0 && t2 == ep && b[t2] == 0 && b[r * 8 + nf] == mk(them, PAWN)) {
            add(sq, t2);
          }
        }
      } else if (t == KNIGHT || t == KING) {
        U64 m = t == KNIGHT ? knightM[sq] : kingM[sq];
        while (m) {
          int to = poplsb(m);
          if (b[to] == 0) { if (!caps) add(sq, to); }
          else if (colorOf(b[to]) == them) add(sq, to);
        }
      } else {
        for (int d = 0; d < 4; d++) {
          for (int pass = 0; pass < 2; pass++) {
            if (pass == 0 && t == ROOK) continue;   // เรือ: ทิศตรงเท่านั้น
            if (pass == 1 && t == BISHOP) continue; // บิชอป: ทแยงเท่านั้น
            const int(*dd)[2] = pass == 0 ? DIAG : ORTH;
            int rr = r + dd[d][0], ff = f + dd[d][1];
            while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) {
              int to = rr * 8 + ff;
              if (b[to] == 0) { if (!caps) add(sq, to); }
              else { if (colorOf(b[to]) == them) add(sq, to); break; }
              rr += dd[d][0]; ff += dd[d][1];
            }
          }
        }
      }
    }
    if (!caps) {  // castling
      if (us == WHITE) {
        if ((castle & 1) && b[4] == mk(0, KING) && b[7] == mk(0, ROOK) && !b[5] && !b[6] &&
            !attacked(4, 1) && !attacked(5, 1) && !attacked(6, 1)) add(4, 6);
        if ((castle & 2) && b[4] == mk(0, KING) && b[0] == mk(0, ROOK) && !b[3] && !b[2] && !b[1] &&
            !attacked(4, 1) && !attacked(3, 1) && !attacked(2, 1)) add(4, 2);
      } else {
        if ((castle & 4) && b[60] == mk(1, KING) && b[63] == mk(1, ROOK) && !b[61] && !b[62] &&
            !attacked(60, 0) && !attacked(61, 0) && !attacked(62, 0)) add(60, 62);
        if ((castle & 8) && b[60] == mk(1, KING) && b[56] == mk(1, ROOK) && !b[59] && !b[58] && !b[57] &&
            !attacked(60, 0) && !attacked(59, 0) && !attacked(58, 0)) add(60, 58);
      }
    }
    return n;
  }

  int genLegal(Move* out) const {
    Move tmp[256];
    int n = gen(tmp, false), k = 0;
    for (int i = 0; i < n; i++) {
      Pos nx = *this;
      nx.make(tmp[i]);
      if (!nx.attacked(nx.ksq[side], side ^ 1)) out[k++] = tmp[i];
    }
    return k;
  }
  int legalCount() const { Move ml[256]; return genLegal(ml); }
};

static inline bool isCap(const Pos& p, Move m) {
  int from = m & 63, to = (m >> 6) & 63;
  return p.b[to] != 0 || (typeOf(p.b[from]) == PAWN && (from & 7) != (to & 7));
}
static string sqStr(int s) { string r; r += char('a' + (s & 7)); r += char('1' + (s >> 3)); return r; }
static string mvStr(Move m) {
  string s = sqStr(m & 63) + sqStr((m >> 6) & 63);
  int pr = (m >> 12) & 7;
  if (pr) s += (pr == KNIGHT ? 'n' : pr == BISHOP ? 'b' : pr == ROOK ? 'r' : 'q');
  return s;
}

static bool insufficient(const Pos& p) {
  int minors = 0;
  for (int s = 0; s < 64; s++) {
    int t = typeOf(p.b[s]);
    if (!p.b[s]) continue;
    if (t == PAWN || t == ROOK || t == QUEEN) return false;
    if (t == KNIGHT || t == BISHOP) minors++;
  }
  return minors <= 1;
}

static string toSAN(const Pos& p, Move m) {
  int from = m & 63, to = (m >> 6) & 63, pr = (m >> 12) & 7;
  int pc = p.b[from], t = typeOf(pc);
  string s;
  if (t == KING && abs(to - from) == 2) s = to > from ? "O-O" : "O-O-O";
  else {
    bool cap = isCap(p, m);
    if (t == PAWN) {
      if (cap) { s += char('a' + (from & 7)); s += 'x'; }
    } else {
      s += " PNBRQK"[t];
      Move ml[256]; int n = p.genLegal(ml);
      bool amb = false, sameF = false, sameR = false;
      for (int i = 0; i < n; i++) {
        int f2 = ml[i] & 63;
        if (ml[i] == m || ((ml[i] >> 6) & 63) != to || p.b[f2] != pc) continue;
        amb = true;
        if ((f2 & 7) == (from & 7)) sameF = true;
        if ((f2 >> 3) == (from >> 3)) sameR = true;
      }
      if (amb) {
        if (!sameF) s += char('a' + (from & 7));
        else if (!sameR) s += char('1' + (from >> 3));
        else { s += char('a' + (from & 7)); s += char('1' + (from >> 3)); }
      }
      if (cap) s += 'x';
    }
    if (!(t == KING && abs(to - from) == 2)) s += sqStr(to);
    if (pr) { s += '='; s += " PNBRQK"[pr]; }
  }
  Pos nx = p; nx.make(m);
  if (nx.inCheck()) s += nx.legalCount() == 0 ? "#" : "+";
  return s;
}

static string normSan(string s) {
  string o;
  for (char c : s) {
    if (c == '+' || c == '#' || c == '!' || c == '?' || c == '=' || c == 'x' || c == ':' || c == '-') continue;
    o += (c == '0') ? 'O' : c;
  }
  return o;
}

static Move parseMove(const Pos& p, string s) {
  while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
  while (!s.empty() && isspace((unsigned char)s[0])) s.erase(0, 1);
  if (s.empty()) return 0;
  Move ml[256]; int n = p.genLegal(ml);
  string lo = s;
  for (auto& c : lo) c = (char)tolower(c);
  for (int i = 0; i < n; i++) if (mvStr(ml[i]) == lo) return ml[i];
  string ns = normSan(s);
  if (ns.find_first_not_of("oO") == string::npos) for (auto& c : ns) c = 'O';
  vector<string> cand = {ns};
  if (!ns.empty() && string("nbrqk").find(ns[0]) != string::npos && ns.size() >= 3) {
    string u = ns; u[0] = (char)toupper(u[0]); cand.push_back(u);
  }
  for (auto& c : cand)
    for (int i = 0; i < n; i++) if (normSan(toSAN(p, ml[i])) == c) return ml[i];
  return 0;
}

static uint64_t perft(const Pos& p, int d) {
  if (d == 0) return 1;
  Move ml[256]; int n = p.genLegal(ml);
  if (d == 1) return n;
  uint64_t t = 0;
  for (int i = 0; i < n; i++) { Pos nx = p; nx.make(ml[i]); t += perft(nx, d - 1); }
  return t;
}

// ============================================================================
//  ข้อมูลตำแหน่ง (bitboard) สำหรับประเมิน
// ============================================================================
struct Info { U64 pc[2][7]; U64 occ[2]; U64 atk[2][7]; };

static inline U64 slideAtt(int sq, const int dirs[4][2], U64 occAll) {
  U64 m = 0; int r0 = sq >> 3, f0 = sq & 7;
  for (int d = 0; d < 4; d++) {
    int r = r0 + dirs[d][0], f = f0 + dirs[d][1];
    while (r >= 0 && r < 8 && f >= 0 && f < 8) {
      int s = r * 8 + f; m |= bit(s);
      if (occAll & bit(s)) break;
      r += dirs[d][0]; f += dirs[d][1];
    }
  }
  return m;
}

static void computeInfo(const Pos& p, Info& I) {
  memset(&I, 0, sizeof I);
  for (int s = 0; s < 64; s++) {
    int pc = p.b[s];
    if (!pc) continue;
    I.pc[colorOf(pc)][typeOf(pc)] |= bit(s);
    I.occ[colorOf(pc)] |= bit(s);
  }
  U64 all = I.occ[0] | I.occ[1];
  for (int c = 0; c < 2; c++)
    for (int t = 1; t <= 6; t++) {
      U64 x = I.pc[c][t];
      while (x) {
        int sq = poplsb(x); U64 m = 0;
        switch (t) {
          case PAWN: m = pawnAtk[c][sq]; break;
          case KNIGHT: m = knightM[sq]; break;
          case BISHOP: m = slideAtt(sq, DIAG, all); break;
          case ROOK: m = slideAtt(sq, ORTH, all); break;
          case QUEEN: m = slideAtt(sq, DIAG, all) | slideAtt(sq, ORTH, all); break;
          case KING: m = kingM[sq]; break;
        }
        I.atk[c][t] |= m; I.atk[c][0] |= m;
      }
    }
}

// ============================================================================
//  "ตัวแปรที่ไม่รู้จัก" (Unknown Variables): ยีน = สูตรนับความสัมพันธ์บนกระดาน
// ============================================================================
static const int NKINDS = 12;
struct Gene {
  int kind = 0, t = 0, aux = 0, f1 = 0, f2 = 7, r1 = 0, r2 = 7;
  U64 mask = ~0ULL;
};
struct Genome {
  Gene a, b;          // ถ้า combo: ค่า = diff(a) * total(b)  (b = บริบท)
  bool combo = false;
  float scale = 1, wmg = 0, weg = 0;
};

static U64 rectMask(int f1, int f2, int r1, int r2) {
  U64 m = 0;
  for (int r = r1; r <= r2; r++) for (int f = f1; f <= f2; f++) m |= bit(r * 8 + f);
  return m;
}
static void fixGene(Gene& g) {
  g.kind = max(0, min(NKINDS - 1, g.kind));
  g.t = max(0, min(6, g.t)); g.aux = max(0, min(6, g.aux));
  g.f1 = max(0, min(7, g.f1)); g.f2 = max(g.f1, min(7, g.f2));
  g.r1 = max(0, min(7, g.r1)); g.r2 = max(g.r1, min(7, g.r2));
  if ((g.kind == 0 || g.kind == 6 || g.kind == 7 || g.kind == 2 || g.kind == 9) && g.t == 0) g.t = 1;
  if (g.kind == 2 && g.aux == 0) g.aux = 1;
  if (g.kind == 3 && (g.aux < 1 || g.aux > 3)) g.aux = max(1, min(3, g.aux));
  g.mask = rectMask(g.f1, g.f2, g.r1, g.r2);
}

static int geneF(const Gene& g, const Info& I, int c) {
  int o = c ^ 1;
  U64 m = c == 0 ? g.mask : __builtin_bswap64(g.mask);  // พลิกแถวสำหรับฝั่งดำ
  switch (g.kind) {
    case 0: return popc(I.pc[c][g.t] & m);
    case 1: return popc(I.atk[c][g.t] & m);
    case 2: return popc(I.atk[c][g.t] & I.pc[o][g.aux]);
    case 3: { if (!I.pc[o][KING]) return 0; return popc(I.atk[c][g.t] & zoneM[lsb(I.pc[o][KING])][g.aux]); }
    case 4: return popc(I.atk[c][g.t] & ~I.occ[c] & ~I.atk[o][PAWN] & m);
    case 5: { U64 x = I.pc[c][PAWN] & m; int n = 0; while (x) { int s = poplsb(x); if (!(passedM[c][s] & I.pc[o][PAWN])) n++; } return n; }
    case 6: return popc(I.pc[c][g.t] & ~I.atk[c][0] & m);
    case 7: return popc(I.pc[c][g.t] & I.atk[o][0] & ~I.atk[c][0] & m);
    case 8: return popc(I.pc[c][PAWN] & I.atk[c][PAWN] & m);
    case 9: { U64 x = I.pc[c][g.t] & m; int n = 0;
      while (x) { int s = poplsb(x); U64 fm = fileM[s & 7];
        if (!(fm & I.pc[c][PAWN]) && (g.aux == 0 || !(fm & I.pc[o][PAWN]))) n++; }
      return n; }
    case 10: return popc(I.atk[c][0] & ~I.atk[o][0] & m);
    case 11: return popc(I.atk[c][0] & I.atk[o][0] & m);
  }
  return 0;
}
static double geneValue(const Genome& g, const Info& I) {
  double d = geneF(g.a, I, 0) - geneF(g.a, I, 1);
  if (!g.combo) return d;
  return d * (geneF(g.b, I, 0) + geneF(g.b, I, 1));
}

static const char* PN[7] = {"ตัวหมากใดๆ", "เบี้ย", "ม้า", "บิชอป", "เรือ", "ควีน", "คิง"};
static string regionStr(const Gene& g) {
  if (g.f1 == 0 && g.f2 == 7 && g.r1 == 0 && g.r2 == 7) return "ทั้งกระดาน";
  string s = "ช่อง ไฟล์ ";
  s += char('a' + g.f1); s += '-'; s += char('a' + g.f2);
  s += " แถว " + to_string(g.r1 + 1) + "-" + to_string(g.r2 + 1) + " (นับจากฝั่งตัวเอง)";
  return s;
}
static string geneDesc(const Gene& g) {
  string r = regionStr(g);
  switch (g.kind) {
    case 0: return string("จำนวน") + PN[g.t] + "ของเรา ใน " + r;
    case 1: return string("จำนวนช่องใน ") + r + " ที่" + PN[g.t] + "ของเราคุมอยู่";
    case 2: return string("จำนวน") + PN[g.aux] + "ศัตรูที่" + PN[g.t] + "ของเราโจมตีอยู่";
    case 3: return string("จำนวนช่องรอบคิงศัตรู (รัศมี ") + to_string(g.aux) + ") ที่" + PN[g.t] + "ของเราคุม";
    case 4: return string("จำนวนช่องปลอดภัยใน ") + r + " ที่" + PN[g.t] + "ของเราไปได้ (ไม่โดนเบี้ยศัตรูคุม)";
    case 5: return "จำนวนเบี้ยทะลุ (passed) ของเราใน " + r;
    case 6: return string("จำนวน") + PN[g.t] + "ของเราใน " + r + " ที่ไม่มีหมากป้องกัน";
    case 7: return string("จำนวน") + PN[g.t] + "ของเราใน " + r + " ที่ถูกโจมตีและไม่มีใครป้องกัน";
    case 8: return "จำนวนเบี้ยของเราใน " + r + " ที่มีเบี้ยด้วยกันป้องกัน (โซ่เบี้ย)";
    case 9: return string("จำนวน") + PN[g.t] + "ของเราใน " + r + (g.aux ? " บนไฟล์เปิดเต็ม" : " บนไฟล์ที่ไม่มีเบี้ยเรา");
    case 10: return "จำนวนช่องใน " + r + " ที่เราคุมแต่ศัตรูไม่คุม (พื้นที่)";
    case 11: return "จำนวนช่องใน " + r + " ที่ทั้งสองฝ่ายคุม (จุดปะทะ)";
  }
  return "?";
}
static string genomeDesc(const Genome& g) {
  if (!g.combo) return "(เรา - ศัตรู) ของ " + geneDesc(g.a);
  return "[(เรา - ศัตรู) ของ " + geneDesc(g.a) + "] x [ผลรวมสองฝ่ายของ " + geneDesc(g.b) + "]";
}

// ============================================================================
//  Brain = ประเมินพื้นฐาน (วัสดุ + ตำแหน่ง) + ตัวแปรที่ค้นพบ
// ============================================================================
struct Brain {
  vector<Genome> gs;

  static void baseTerms(const Pos& p, int& mg, int& eg, int& phase) {
    static const int VMG[7] = {0, 82, 337, 365, 477, 1025, 0};
    static const int VEG[7] = {0, 94, 281, 297, 512, 936, 0};
    mg = eg = 0; phase = 0;
    for (int s = 0; s < 64; s++) {
      int pc = p.b[s];
      if (!pc) continue;
      int c = colorOf(pc), t = typeOf(pc), f = s & 7, r = s >> 3;
      if (c == BLACK) r = 7 - r;
      double dist = fabs(f - 3.5) + fabs(r - 3.5);
      int pm = 0, pe = 0;
      switch (t) {
        case PAWN: pm = (r - 1) * 4 + ((f == 3 || f == 4) ? 6 : 0); pe = (r - 1) * 10; break;
        case KNIGHT: pm = pe = (int)(36 - 9 * dist); phase += 1; break;
        case BISHOP: pm = pe = (int)(20 - 4 * dist); phase += 1; break;
        case ROOK: pm = r == 6 ? 12 : 0; pe = 0; phase += 2; break;
        case QUEEN: pm = pe = (int)(8 - 2 * dist); phase += 4; break;
        case KING: pm = (r == 0 ? ((f <= 2 || f >= 6) ? 25 : 5) : -25 * r); pe = (int)(24 - 6 * dist); break;
      }
      int vm = VMG[t] + pm, ve = VEG[t] + pe;
      if (c == WHITE) { mg += vm; eg += ve; } else { mg -= vm; eg -= ve; }
    }
    if (phase > 24) phase = 24;
  }

  double evalWhite(const Pos& p) const {
    int mg, eg, ph; baseTerms(p, mg, eg, ph);
    double a = ph / 24.0, b = 1 - a;
    double e = mg * a + eg * b;
    if (!gs.empty()) {
      Info I; computeInfo(p, I);
      for (auto& g : gs) { double v = geneValue(g, I) / g.scale; e += v * (g.wmg * a + g.weg * b); }
    }
    return e;
  }
  // คะแนนจากมุมมองฝั่งที่จะเดิน
  int evaluate(const Pos& p) const {
    double e = evalWhite(p);
    if (e > 15000) e = 15000;
    if (e < -15000) e = -15000;
    int s = (int)lround(e);
    return (p.side == WHITE ? s : -s) + 8;
  }

  bool save(const string& fn) const {
    ofstream o(fn);
    if (!o) return false;
    o << "UNCHESS_BRAIN 1\n" << gs.size() << "\n";
    auto wg = [&](const Gene& g) { o << g.kind << ' ' << g.t << ' ' << g.aux << ' ' << g.f1 << ' ' << g.f2 << ' ' << g.r1 << ' ' << g.r2 << ' '; };
    for (auto& g : gs) { o << (g.combo ? 1 : 0) << ' '; wg(g.a); wg(g.b); o << g.scale << ' ' << g.wmg << ' ' << g.weg << "\n"; }
    return (bool)o;
  }
  bool load(const string& fn) {
    ifstream in(fn);
    if (!in) return false;
    string tag; int ver = 0; size_t n = 0;
    if (!(in >> tag >> ver >> n) || tag != "UNCHESS_BRAIN" || n > 1000) return false;
    vector<Genome> v;
    auto rg = [&](Gene& g) { return (bool)(in >> g.kind >> g.t >> g.aux >> g.f1 >> g.f2 >> g.r1 >> g.r2); };
    for (size_t i = 0; i < n; i++) {
      Genome g; int cb;
      if (!(in >> cb) || !rg(g.a) || !rg(g.b) || !(in >> g.scale >> g.wmg >> g.weg)) return false;
      g.combo = cb != 0; fixGene(g.a); fixGene(g.b);
      if (!(g.scale > 1e-9f) || !isfinite(g.wmg) || !isfinite(g.weg)) return false;
      v.push_back(g);
    }
    gs = v;
    return true;
  }
};

// ============================================================================
//  การค้นหา (alpha-beta + TT + quiescence + null move + LMR)
// ============================================================================
struct Limits { int depth = 64; long long nodes = 0; int hardMs = 0, softMs = 0; };
struct TTE { U64 key = 0; int move = 0; int score = 0; short depth = 0; unsigned char flag = 0; };
static const int PV_VAL[7] = {0, 100, 320, 330, 500, 900, 20000};

class Searcher {
 public:
  const Brain* br;
  vector<TTE> tt; U64 ttmask;
  atomic<bool>* ext = nullptr;
  bool stopped = false; long long nodes = 0; int completed = 0;
  Limits lim; chrono::steady_clock::time_point t0;
  Move killers[MAXPLY + 4][2]; int histo[2][64][64];
  vector<U64> rs; int rbase = 0; Move rootBest = 0;

  Searcher(const Brain* b, int log2 = 18) : br(b) {
    tt.assign((size_t)1 << log2, TTE()); ttmask = ((U64)1 << log2) - 1;
  }
  void clearTT() { fill(tt.begin(), tt.end(), TTE()); }
  long long elapsedMs() const { return chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - t0).count(); }
  void timeCheck() {
    if (completed < 1) return;
    if (ext && ext->load()) { stopped = true; return; }
    if (lim.nodes && nodes >= lim.nodes) { stopped = true; return; }
    if (lim.hardMs && elapsedMs() >= lim.hardMs) stopped = true;
  }
  static int toTT(int s, int ply) { if (s > MATE - MAXPLY) return s + ply; if (s < -MATE + MAXPLY) return s - ply; return s; }
  static int fromTT(int s, int ply) { if (s > MATE - MAXPLY) return s - ply; if (s < -MATE + MAXPLY) return s + ply; return s; }
  bool isRep(const Pos& p, int ply) const {
    int idx = rbase + ply;
    for (int j = idx - 2; j >= 0 && j >= idx - p.half; j -= 2) if (rs[j] == p.key) return true;
    return false;
  }
  static bool nonPawn(const Pos& p, int c) {
    for (int s = 0; s < 64; s++) { int pc = p.b[s]; if (pc && colorOf(pc) == c) { int t = typeOf(pc); if (t >= KNIGHT && t <= QUEEN) return true; } }
    return false;
  }
  int capScore(const Pos& p, Move m) const {
    int from = m & 63, to = (m >> 6) & 63;
    int victim = p.b[to] ? typeOf(p.b[to]) : PAWN;
    return PV_VAL[victim] * 16 - PV_VAL[typeOf(p.b[from])] / 16;
  }

  int qsearch(const Pos& p, int alpha, int beta, int ply) {
    if (stopped) return 0;
    if ((++nodes & 1023) == 0) timeCheck();
    if (stopped) return 0;
    if (insufficient(p)) return 0;
    if (ply >= MAXPLY - 1) return br->evaluate(p);
    bool inchk = p.inCheck();
    int best = -INF;
    if (!inchk) { best = br->evaluate(p); if (best >= beta) return best; if (best > alpha) alpha = best; }
    Move ml[256]; int sc[256];
    int n = p.gen(ml, !inchk);
    for (int i = 0; i < n; i++) sc[i] = isCap(p, ml[i]) ? capScore(p, ml[i]) + 100000 : (((ml[i] >> 12) & 7) ? 90000 : 0);
    int legal = 0;
    for (int i = 0; i < n; i++) {
      int bi = i;
      for (int j = i + 1; j < n; j++) if (sc[j] > sc[bi]) bi = j;
      swap(ml[i], ml[bi]); swap(sc[i], sc[bi]);
      Pos nx = p; nx.make(ml[i]);
      if (nx.attacked(nx.ksq[p.side], p.side ^ 1)) continue;
      legal++;
      int s = -qsearch(nx, -beta, -alpha, ply + 1);
      if (stopped) return 0;
      if (s > best) { best = s; if (s > alpha) { alpha = s; if (alpha >= beta) break; } }
    }
    if (inchk && legal == 0) return -MATE + ply;
    return best;
  }

  int search(const Pos& p, int depth, int alpha, int beta, int ply, bool nullOk) {
    if (stopped) return 0;
    if ((++nodes & 1023) == 0) timeCheck();
    if (stopped) return 0;
    bool inchk = p.inCheck();
    if (ply > 0) {
      if (isRep(p, ply) || insufficient(p)) return 0;
      if (p.half >= 100) { if (!inchk) return 0; return p.legalCount() > 0 ? 0 : -MATE + ply; }
    }
    if (ply >= MAXPLY - 1) return br->evaluate(p);
    if (inchk) depth++;
    if (depth <= 0) return qsearch(p, alpha, beta, ply);
    rs[rbase + ply] = p.key;
    bool pv = beta - alpha > 1;
    int origAlpha = alpha;
    TTE& e = tt[p.key & ttmask];
    Move ttm = 0;
    if (e.key == p.key) {
      ttm = e.move;
      if (ply > 0 && e.depth >= depth) {
        int s = fromTT(e.score, ply);
        if (e.flag == 0) return s;
        if (e.flag == 1 && s >= beta) return s;
        if (e.flag == 2 && s <= alpha) return s;
      }
    }
    int se = inchk ? -INF : br->evaluate(p);
    if (!pv && !inchk && ply > 0 && depth <= 3 && abs(beta) < MATE - 1000 && se - 110 * depth >= beta) return se;
    if (!pv && !inchk && nullOk && depth >= 3 && se >= beta && nonPawn(p, p.side)) {
      Pos nx = p; nx.makeNull();
      int R = 2 + (depth >= 6);
      int s = -search(nx, depth - 1 - R, -beta, -beta + 1, ply + 1, false);
      if (stopped) return 0;
      if (s >= beta) return s >= MATE - MAXPLY ? beta : s;
    }
    Move ml[256]; int sc[256];
    int n = p.gen(ml, false);
    int us = p.side;
    for (int i = 0; i < n; i++) {
      Move m = ml[i];
      if (m == ttm) sc[i] = 1000000;
      else if (isCap(p, m)) sc[i] = 100000 + capScore(p, m);
      else if ((m >> 12) & 7) sc[i] = 95000;
      else if (m == killers[ply][0]) sc[i] = 90000;
      else if (m == killers[ply][1]) sc[i] = 80000;
      else sc[i] = histo[us][m & 63][(m >> 6) & 63];
    }
    int best = -INF, legal = 0; Move bestMove = 0;
    for (int i = 0; i < n; i++) {
      int bi = i;
      for (int j = i + 1; j < n; j++) if (sc[j] > sc[bi]) bi = j;
      swap(ml[i], ml[bi]); swap(sc[i], sc[bi]);
      Move m = ml[i];
      Pos nx = p; nx.make(m);
      if (nx.attacked(nx.ksq[us], us ^ 1)) continue;
      legal++;
      bool capm = isCap(p, m), promo = ((m >> 12) & 7) != 0;
      int nd = depth - 1, s;
      if (legal == 1) s = -search(nx, nd, -beta, -alpha, ply + 1, true);
      else {
        int red = 0;
        if (depth >= 3 && legal > 4 && !inchk && !capm && !promo && !nx.inCheck()) {
          red = 1 + (legal > 10) + (depth > 6);
          red = min(red, nd - 1);
        }
        s = -search(nx, nd - red, -alpha - 1, -alpha, ply + 1, true);
        if (s > alpha && red > 0) s = -search(nx, nd, -alpha - 1, -alpha, ply + 1, true);
        if (s > alpha && s < beta) s = -search(nx, nd, -beta, -alpha, ply + 1, true);
      }
      if (stopped) return 0;
      if (s > best) {
        best = s; bestMove = m;
        if (ply == 0) rootBest = m;
        if (s > alpha) {
          alpha = s;
          if (alpha >= beta) {
            if (!capm && !promo) {
              if (killers[ply][0] != m) { killers[ply][1] = killers[ply][0]; killers[ply][0] = m; }
              int& h = histo[us][m & 63][(m >> 6) & 63];
              h += depth * depth; if (h > 70000) h = 70000;
            }
            break;
          }
        }
      }
    }
    if (legal == 0) return inchk ? -MATE + ply : 0;
    e.key = p.key; e.move = bestMove; e.score = toTT(best, ply); e.depth = (short)depth;
    e.flag = best <= origAlpha ? 2 : (best >= beta ? 1 : 0);
    return best;
  }

  string pvString(const Pos& root, int maxLen) {
    string s; Pos q = root;
    for (int i = 0; i < maxLen; i++) {
      TTE& e = tt[q.key & ttmask];
      if (e.key != q.key || !e.move) break;
      Move ml[256]; int n = q.genLegal(ml); bool ok = false;
      for (int k = 0; k < n; k++) if (ml[k] == e.move) ok = true;
      if (!ok) break;
      s += mvStr(e.move) + " ";
      q.make(e.move);
    }
    return s;
  }

  // คืนตาเดินที่ "ถูกกฎแน่นอน" เสมอ (ถ้ามีตาเดินถูกกฎ)
  Move think(const Pos& root, const vector<U64>& hist, const Limits& L, int* outScore, bool verbose) {
    lim = L; stopped = false; nodes = 0; completed = 0; t0 = chrono::steady_clock::now();
    memset(killers, 0, sizeof killers); memset(histo, 0, sizeof histo);
    rs = hist; if (rs.empty()) rs.push_back(root.key);
    rbase = (int)rs.size() - 1; rs[rbase] = root.key;
    rs.resize(rs.size() + MAXPLY + 8);
    Move legal[256]; int n = root.genLegal(legal);
    if (outScore) *outScore = 0;
    if (n == 0) return 0;
    Move best = legal[0]; int bestScore = 0;
    if (n == 1) { if (outScore) *outScore = br->evaluate(root); return best; }
    for (int d = 1; d <= min(L.depth, MAXPLY - 4); d++) {
      rootBest = 0;
      int s = search(root, d, -INF, INF, 0, false);
      if (stopped) {
        if (rootBest) { bool ok = false; for (int i = 0; i < n; i++) if (legal[i] == rootBest) ok = true; if (ok) best = rootBest; }
        break;
      }
      if (rootBest) best = rootBest;
      bestScore = s; completed = d;
      if (verbose) {
        printf("info depth %d score ", d);
        if (abs(s) > MATE - MAXPLY) printf("mate %d", (s > 0 ? (MATE - s + 1) / 2 : -(MATE + s + 1) / 2));
        else printf("cp %d", s);
        printf(" nodes %lld time %lld pv %s\n", nodes, elapsedMs(), pvString(root, d).c_str());
        fflush(stdout);
      }
      if (abs(s) > MATE - MAXPLY && d >= 2) break;
      if (L.softMs && elapsedMs() >= L.softMs) break;
    }
    // ตรวจสอบสุดท้าย: ต้องอยู่ในรายการตาเดินที่ถูกกฎเท่านั้น
    bool ok = false;
    for (int i = 0; i < n; i++) if (legal[i] == best) ok = true;
    if (!ok) best = legal[0];
    if (outScore) *outScore = bestScore;
    return best;
  }
};

// ============================================================================
//  ระบบ self-play + ค้นพบตัวแปรใหม่
// ============================================================================
struct Sample { Pos p; float y; };
struct Rec { Pos p; int sw; };

static double sigm(double x) { return 1.0 / (1.0 + exp(-x / 400.0)); }

static bool randomOpening(mt19937& rng, int plies, Pos& out, vector<U64>& hist) {
  static Brain basic;
  for (int tries = 0; tries < 50; tries++) {
    Pos p; p.setFen(STARTFEN); hist.assign(1, p.key);
    bool bad = false;
    for (int i = 0; i < plies; i++) {
      Move ml[256]; int n = p.genLegal(ml);
      if (n == 0) { bad = true; break; }
      p.make(ml[rng() % n]); hist.push_back(p.key);
    }
    if (bad || p.legalCount() == 0) continue;
    if (fabs(basic.evalWhite(p)) > 250) continue;
    out = p; return true;
  }
  return false;
}

// เล่นจบหนึ่งเกม; คืน 1 = ขาวชนะ, 0 = เสมอ, -1 = ดำชนะ
static int playGame(Searcher* S[2], const Pos& start, vector<U64> hist, const Limits& L, mt19937& rng,
                    vector<Rec>* rec, int randPct) {
  Pos p = start; int ply = 0, winS = 0, loseS = 0;
  while (true) {
    Move ml[256]; int n = p.genLegal(ml);
    if (n == 0) return p.inCheck() ? (p.side == WHITE ? -1 : 1) : 0;
    if (p.half >= 100 || insufficient(p) || ply >= 300) return 0;
    int cnt = 0; for (U64 k : hist) if (k == p.key) cnt++;
    if (cnt >= 3) return 0;
    int sc = 0;
    Move m = S[p.side]->think(p, hist, L, &sc, false);
    bool rnd = false;
    if (randPct > 0 && ply < 24 && (int)(rng() % 100) < randPct) { m = ml[rng() % n]; rnd = true; }
    int sw = p.side == WHITE ? sc : -sc;
    if (rec && !rnd && !p.inCheck() && ply >= 2 && abs(sc) < 1500 && !isCap(p, m)) rec->push_back({p, sw});
    if (!rnd) {
      if (sw >= 1200) { winS++; loseS = 0; } else if (sw <= -1200) { loseS++; winS = 0; } else winS = loseS = 0;
      if (winS >= 6) return 1;
      if (loseS >= 6) return -1;
    }
    p.make(m); hist.push_back(p.key); ply++;
  }
}

static int hwThreads() { int t = (int)thread::hardware_concurrency(); return max(1, min(8, t)); }

static vector<Sample> generateData(const Brain& br, int targetN, int depth, int threads, double lam, unsigned seed) {
  vector<Sample> out; mutex mu; atomic<int> games(0);
  auto worker = [&](int id) {
    mt19937 rng(seed * 1000003u + id * 7919u);
    Searcher A(&br, 17); Searcher* S[2] = {&A, &A};
    Limits L; L.depth = depth;
    vector<Rec> rec;
    while (true) {
      { lock_guard<mutex> lk(mu); if ((int)out.size() >= targetN) break; }
      Pos st; vector<U64> h;
      if (!randomOpening(rng, 4 + rng() % 5, st, h)) continue;
      A.clearTT(); rec.clear();
      int res = playGame(S, st, h, L, rng, &rec, 6);
      lock_guard<mutex> lk(mu);
      double r = (res + 1) / 2.0;
      for (auto& x : rec) out.push_back({x.p, (float)(lam * r + (1 - lam) * sigm(x.sw))});
      games++;
      printf("\r  [เก็บข้อมูล] เกม %d  ตำแหน่ง %d/%d   ", games.load(), (int)out.size(), targetN);
      fflush(stdout);
    }
  };
  vector<thread> th;
  for (int i = 0; i < threads; i++) th.emplace_back(worker, i);
  for (auto& t : th) t.join();
  printf("\n");
  return out;
}

// แมตช์: A ปะทะ B (สลับสี) -> คืนคะแนนของ A (0..1)
static double runMatch(const Brain& A, const Brain& B, int pairs, int depth, int threads, unsigned seed, int& W, int& D, int& Lz) {
  W = D = Lz = 0; mutex mu; atomic<int> next(0);
  auto worker = [&](int id) {
    mt19937 rng(seed * 2654435761u + id);
    Searcher sa(&A, 16), sb(&B, 16); Limits L; L.depth = depth;
    while (true) {
      int k = next++;
      if (k >= pairs) break;
      mt19937 orng(seed * 977u + k);
      Pos st; vector<U64> h;
      if (!randomOpening(orng, 6 + (k % 3), st, h)) continue;
      for (int side = 0; side < 2; side++) {
        sa.clearTT(); sb.clearTT();
        Searcher* S[2] = {side == 0 ? &sa : &sb, side == 0 ? &sb : &sa};
        int res = playGame(S, st, h, L, rng, nullptr, 0);
        int fromA = side == 0 ? res : -res;
        lock_guard<mutex> lk(mu);
        if (fromA > 0) W++; else if (fromA < 0) Lz++; else D++;
        printf("\r  [แมตช์] %d-%d-%d (ชนะ-เสมอ-แพ้)   ", W, D, Lz); fflush(stdout);
      }
    }
  };
  vector<thread> th;
  for (int i = 0; i < threads; i++) th.emplace_back(worker, i);
  for (auto& t : th) t.join();
  printf("\n");
  int tot = W + D + Lz;
  return tot ? (W + 0.5 * D) / tot : 0.5;
}

static Gene randGene(mt19937& rng) {
  auto R = [&](int lo, int hi) { return lo + (int)(rng() % (hi - lo + 1)); };
  Gene g; g.kind = R(0, NKINDS - 1);
  switch (g.kind) {
    case 0: g.t = R(1, 6); break;
    case 1: g.t = R(0, 6); break;
    case 2: g.t = R(1, 6); g.aux = R(1, 6); break;
    case 3: g.t = R(0, 6); g.aux = R(1, 3); break;
    case 4: g.t = R(0, 6); break;
    case 6: case 7: g.t = R(1, 6); break;
    case 9: g.t = R(2, 5); g.aux = R(0, 1); break;
  }
  if (rng() % 2) { g.f1 = 0; g.f2 = 7; g.r1 = 0; g.r2 = 7; }
  else { g.f1 = R(0, 7); g.f2 = R(g.f1, 7); g.r1 = R(0, 7); g.r2 = R(g.r1, 7); }
  fixGene(g); return g;
}
static Genome randGenome(mt19937& rng, const vector<Genome>& pool) {
  Genome g; int roll = rng() % 100;
  if (!pool.empty() && roll < 30) {  // กลายพันธุ์จากยีนที่มีอยู่
    g = pool[rng() % pool.size()]; g.wmg = g.weg = 0; g.scale = 1;
    Gene& t = (g.combo && rng() % 2) ? g.b : g.a;
    Gene r = randGene(rng);
    switch (rng() % 4) {
      case 0: t.t = r.t; break; case 1: t.aux = r.aux; break;
      case 2: t.f1 = r.f1; t.f2 = r.f2; break; default: t.r1 = r.r1; t.r2 = r.r2; break;
    }
    fixGene(t); return g;
  }
  g.a = randGene(rng);
  if (roll < 50) { g.combo = true; g.b = randGene(rng); }
  return g;
}
static string geneKey(const Genome& g) {
  ostringstream o;
  auto w = [&](const Gene& x) { o << x.kind << ',' << x.t << ',' << x.aux << ',' << x.f1 << ',' << x.f2 << ',' << x.r1 << ',' << x.r2 << ';'; };
  o << g.combo << ';'; w(g.a); if (g.combo) w(g.b);
  return o.str();
}

struct FitSet {
  int N = 0;
  vector<Info> I; vector<float> a, base, y; vector<char> val;
};

static double lossOf(const FitSet& F, const vector<double>& e, int wantVal) {
  double s = 0; int c = 0;
  for (int i = 0; i < F.N; i++) { if ((int)F.val[i] != wantVal) continue; double d = F.y[i] - sigm(e[i]); s += d * d; c++; }
  return c ? s / c : 0;
}

// ปรับน้ำหนักของทุกยีนด้วย Adam (ลด MSE ระหว่างผลคาดการณ์กับผลจริง/คะแนนค้นหาลึก)
static void fitWeights(Brain& br, const FitSet& F, const vector<vector<float>>& raw, int iters) {
  int G = (int)br.gs.size(), N = F.N;
  if (!G) return;
  vector<float> U((size_t)N * G);
  for (int j = 0; j < G; j++) for (int i = 0; i < N; i++) U[(size_t)i * G + j] = raw[j][i] / br.gs[j].scale;
  vector<double> w(2 * G), m1(2 * G, 0), m2(2 * G, 0);
  for (int j = 0; j < G; j++) { w[2 * j] = br.gs[j].wmg; w[2 * j + 1] = br.gs[j].weg; }
  int ntr = 0; for (int i = 0; i < N; i++) if (!F.val[i]) ntr++;
  vector<double> gr(2 * G);
  for (int it = 1; it <= iters; it++) {
    fill(gr.begin(), gr.end(), 0.0);
    for (int i = 0; i < N; i++) {
      if (F.val[i]) continue;
      const float* u = &U[(size_t)i * G];
      double s1 = 0, s2 = 0;
      for (int j = 0; j < G; j++) { s1 += w[2 * j] * u[j]; s2 += w[2 * j + 1] * u[j]; }
      double a = F.a[i], e = F.base[i] + a * s1 + (1 - a) * s2;
      double p = sigm(e), c = -2 * (F.y[i] - p) * p * (1 - p) / 400.0;
      for (int j = 0; j < G; j++) { gr[2 * j] += c * a * u[j]; gr[2 * j + 1] += c * (1 - a) * u[j]; }
    }
    double lr = 2.0 * (1.0 - 0.9 * it / iters);
    for (int k = 0; k < 2 * G; k++) {
      double g = gr[k] / max(1, ntr) + 2e-6 * w[k];
      m1[k] = 0.9 * m1[k] + 0.1 * g; m2[k] = 0.999 * m2[k] + 0.001 * g * g;
      double mh = m1[k] / (1 - pow(0.9, it)), vh = m2[k] / (1 - pow(0.999, it));
      w[k] -= lr * mh / (sqrt(vh) + 1e-12);
      w[k] = max(-300.0, min(300.0, w[k]));
    }
  }
  for (int j = 0; j < G; j++) { br.gs[j].wmg = (float)w[2 * j]; br.gs[j].weg = (float)w[2 * j + 1]; }
}

static void currentEval(const Brain& br, const FitSet& F, vector<vector<float>>& raw, vector<double>& e) {
  int G = (int)br.gs.size();
  raw.assign(G, vector<float>(F.N));
  e.assign(F.N, 0);
  for (int i = 0; i < F.N; i++) {
    double s = F.base[i], a = F.a[i];
    for (int j = 0; j < G; j++) {
      raw[j][i] = (float)geneValue(br.gs[j], F.I[i]);
      s += raw[j][i] / br.gs[j].scale * (br.gs[j].wmg * a + br.gs[j].weg * (1 - a));
    }
    e[i] = s;
  }
}

// ขั้นตอนค้นพบ: คืน true ถ้ามีการเปลี่ยนแปลง (ผ่านการทดสอบกับชุดตรวจสอบแล้ว)
static bool discover(Brain& br, const vector<Sample>& S, int ncand, int maxAdd, mt19937& rng, ofstream& logf) {
  FitSet F; F.N = (int)S.size();
  if (F.N < 500) { printf("  ข้อมูลน้อยเกินไป\n"); return false; }
  F.I.resize(F.N); F.a.resize(F.N); F.base.resize(F.N); F.y.resize(F.N); F.val.resize(F.N);
  for (int i = 0; i < F.N; i++) {
    computeInfo(S[i].p, F.I[i]);
    int mg, eg, ph; Brain::baseTerms(S[i].p, mg, eg, ph);
    F.a[i] = ph / 24.0f; F.base[i] = (float)(mg * F.a[i] + eg * (1 - F.a[i]));
    F.y[i] = S[i].y; F.val[i] = ((i / 40) % 4 == 0) ? 1 : 0;
  }
  vector<vector<float>> raw; vector<double> e;
  currentEval(br, F, raw, e);
  double trB = lossOf(F, e, 0), vaB = lossOf(F, e, 1);
  printf("  ความคลาดเคลื่อนก่อนหน้า: train %.5f  val %.5f\n", trB, vaB);
  vector<double> r(F.N), d(F.N);
  for (int i = 0; i < F.N; i++) { double p = sigm(e[i]); r[i] = F.y[i] - p; d[i] = p * (1 - p); }

  struct Cand { Genome g; double t; vector<float> col; };
  vector<Cand> pass; double maxT = 0, maxV = 0;
  set<string> have; for (auto& g : br.gs) have.insert(geneKey(g));
  for (int c = 0; c < ncand; c++) {
    Genome g = randGenome(rng, br.gs);
    string k = geneKey(g);
    if (have.count(k)) continue;
    have.insert(k);
    vector<float> col(F.N);
    double gt = 0, ht = 0, gv = 0, hv = 0, sum = 0, sq = 0; int nz = 0;
    for (int i = 0; i < F.N; i++) {
      double v = geneValue(g, F.I[i]); col[i] = (float)v;
      sum += v; sq += v * v; if (v != 0) nz++;
      double gg = r[i] * d[i] * v, hh = r[i] * r[i] * d[i] * d[i] * v * v;
      if (F.val[i]) { gv += gg; hv += hh; } else { gt += gg; ht += hh; }
    }
    double mean = sum / F.N, var = sq / F.N - mean * mean;
    if (var < 1e-9 || nz < F.N / 100 || ht <= 0 || hv <= 0) continue;
    double tt = gt / sqrt(ht), tv = gv / sqrt(hv);
    if (fabs(tt) > maxT) { maxT = fabs(tt); maxV = tv * (tt > 0 ? 1 : -1); }
    if (fabs(tt) >= 3.5 && fabs(tv) >= 2.0 && tt * tv > 0) { g.scale = (float)sqrt(var); pass.push_back({g, fabs(tt), col}); }
  }
  printf("  ทดลองทฤษฎีผู้สมัคร %d แบบ ผ่านด่านแรก %d แบบ (t สูงสุด train %.2f, val %.2f)\n", ncand, (int)pass.size(), maxT, maxV);
  sort(pass.begin(), pass.end(), [](const Cand& x, const Cand& y2) { return x.t > y2.t; });

  Brain nb = br; vector<vector<float>> nraw = raw; int added = 0;
  auto corr = [&](const vector<float>& x, const vector<float>& y2) {
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0; int n = 0;
    for (int i = 0; i < F.N; i++) { if (F.val[i]) continue; sx += x[i]; sy += y2[i]; sxx += (double)x[i] * x[i]; syy += (double)y2[i] * y2[i]; sxy += (double)x[i] * y2[i]; n++; }
    double cv = sxy / n - sx / n * sy / n, vx = sxx / n - sx / n * sx / n, vy = syy / n - sy / n * sy / n;
    return (vx > 1e-12 && vy > 1e-12) ? cv / sqrt(vx * vy) : 1.0;
  };
  for (auto& c : pass) {
    if (added >= maxAdd) break;
    bool dup = false;
    for (auto& rc : nraw) if (fabs(corr(c.col, rc)) > 0.9) { dup = true; break; }
    if (dup) continue;
    nb.gs.push_back(c.g); nraw.push_back(c.col); added++;
  }
  if (!added) { printf("  ไม่พบตัวแปรใหม่ที่น่าเชื่อถือรอบนี้\n"); return false; }
  fitWeights(nb, F, nraw, 400);
  vector<double> e2(F.N);
  for (int i = 0; i < F.N; i++) {
    double s = F.base[i], a = F.a[i];
    for (size_t j = 0; j < nb.gs.size(); j++) s += nraw[j][i] / nb.gs[j].scale * (nb.gs[j].wmg * a + nb.gs[j].weg * (1 - a));
    e2[i] = s;
  }
  double trA = lossOf(F, e2, 0), vaA = lossOf(F, e2, 1);
  printf("  หลังเรียนรู้: train %.5f  val %.5f\n", trA, vaA);
  if (!(vaA < vaB - 1e-5)) { printf("  ตัวแปรใหม่ไม่ช่วยในชุดตรวจสอบ -> ทิ้ง\n"); return false; }
  // ทิ้งตัวแปรที่น้ำหนักเล็กจนไม่มีผล
  Brain pr; for (auto& g : nb.gs) if (max(fabs(g.wmg), fabs(g.weg)) >= 0.5f) pr.gs.push_back(g);
  while (pr.gs.size() > 48) {
    size_t wi = 0; float wv = 1e9;
    for (size_t j = 0; j < pr.gs.size(); j++) { float v = fabs(pr.gs[j].wmg) + fabs(pr.gs[j].weg); if (v < wv) { wv = v; wi = j; } }
    pr.gs.erase(pr.gs.begin() + wi);
  }
  for (size_t j = br.gs.size(); j < nb.gs.size(); j++) {
    printf("  ★ ค้นพบ: %s\n      น้ำหนัก (ช่วงกลางเกม/ท้ายเกม): %+.1f / %+.1f เซนติเบี้ย ต่อ 1 SD\n",
           genomeDesc(nb.gs[j]).c_str(), nb.gs[j].wmg, nb.gs[j].weg);
    if (logf) logf << "ค้นพบ: " << genomeDesc(nb.gs[j]) << " | น้ำหนัก " << nb.gs[j].wmg << " / " << nb.gs[j].weg << "\n";
  }
  br = pr;
  return true;
}

static void learnMode(int argc, char** argv, const string& defBrain) {
  int gens = 10, games = 25, depth = 4, npos = 15000, threads = hwThreads(), ncand = 800;
  double accept = 0.52; string brainFile = defBrain; unsigned seed = (unsigned)time(nullptr);
  for (int i = 2; i < argc; i++) {
    string a = argv[i];
    auto nxt = [&]() { return i + 1 < argc ? string(argv[++i]) : string("0"); };
    if (a == "--gens") gens = atoi(nxt().c_str());
    else if (a == "--games") games = atoi(nxt().c_str());   // จำนวน "คู่" เกมในแมตช์ตรวจสอบ
    else if (a == "--depth") depth = atoi(nxt().c_str());
    else if (a == "--pos") npos = atoi(nxt().c_str());
    else if (a == "--threads") threads = max(1, atoi(nxt().c_str()));
    else if (a == "--cands") ncand = atoi(nxt().c_str());
    else if (a == "--accept") accept = atof(nxt().c_str());
    else if (a == "--brain") brainFile = nxt();
    else if (a == "--seed") seed = (unsigned)atoi(nxt().c_str());
  }
  Brain cur; if (cur.load(brainFile)) printf("โหลดสมองเดิม: %s (%d ตัวแปร)\n", brainFile.c_str(), (int)cur.gs.size());
  ofstream logf("unchess_discoveries.txt", ios::app);
  mt19937 rng(seed);
  printf("UnChess Unknown Variable Engine — เริ่มเรียนรู้ (threads=%d, depth=%d)\n", threads, depth);
  for (int g = 1; g <= gens; g++) {
    printf("\n=== รอบที่ %d/%d ===\n", g, gens);
    auto S = generateData(cur, npos, depth, threads, 0.4, rng());
    Brain cand = cur;
    bool ch = discover(cand, S, ncand, 3, rng, logf);
    if (!ch) continue;
    int W, D, L;
    double sc = runMatch(cand, cur, games, depth, threads, rng(), W, D, L);
    double elo = sc <= 0.001 ? -999 : sc >= 0.999 ? 999 : -400 * log10(1 / sc - 1);
    printf("  ผลแมตช์ใหม่ vs เดิม: %.1f%% (ประมาณ %+.0f Elo)\n", sc * 100, elo);
    if (sc >= accept) {
      cur = cand;
      if (cur.save(brainFile)) printf("  ✔ ยอมรับ และบันทึกลง %s (%d ตัวแปร)\n", brainFile.c_str(), (int)cur.gs.size());
      else printf("  ⚠ บันทึกไฟล์ไม่สำเร็จ\n");
    } else printf("  ✘ ไม่ผ่านการทดสอบการเล่นจริง -> ทิ้ง\n");
  }
  printf("\nเสร็จสิ้น. ตัวแปรทั้งหมดที่ใช้อยู่ตอนนี้:\n");
  for (size_t j = 0; j < cur.gs.size(); j++)
    printf(" %2zu) %s  [%+.1f/%+.1f]\n", j + 1, genomeDesc(cur.gs[j]).c_str(), cur.gs[j].wmg, cur.gs[j].weg);
}

// ============================================================================
//  UCI
// ============================================================================
static void uciLoop(Brain& brain, const string& brainFile) {
  Pos pos; pos.setFen(STARTFEN); vector<U64> hist = {pos.key};
  int hashMB = 32;
  auto mkSearcher = [&]() { int lg = 16; while (lg < 24 && ((size_t(1) << (lg + 1)) * sizeof(TTE)) <= (size_t)hashMB * 1024 * 1024) lg++; return new Searcher(&brain, lg); };
  Searcher* S = mkSearcher();
  atomic<bool> stopFlag(false); S->ext = &stopFlag;
  thread worker;
  auto join = [&]() { if (worker.joinable()) worker.join(); };
  string line;
  while (getline(cin, line)) {
    istringstream ss(line); string cmd; ss >> cmd;
    if (cmd == "uci") {
      printf("id name UnChess (Unknown Variable Engine)\nid author UnChess\n");
      printf("option name Hash type spin default 32 min 1 max 1024\n");
      printf("option name BrainFile type string default %s\nuciok\n", brainFile.c_str());
    } else if (cmd == "isready") { join(); printf("readyok\n"); }
    else if (cmd == "ucinewgame") { join(); S->clearTT(); pos.setFen(STARTFEN); hist = {pos.key}; }
    else if (cmd == "setoption") {
      string tok, name, val; ss >> tok; ss >> name; ss >> tok;
      getline(ss, val); while (!val.empty() && val[0] == ' ') val.erase(0, 1);
      join();
      if (name == "Hash") { hashMB = max(1, min(1024, atoi(val.c_str()))); delete S; S = mkSearcher(); S->ext = &stopFlag; }
      else if (name == "BrainFile") { Brain nb; if (nb.load(val)) brain = nb; }
    } else if (cmd == "position") {
      join();
      string tok; ss >> tok; Pos np; bool ok = true;
      if (tok == "startpos") np.setFen(STARTFEN);
      else if (tok == "fen") {
        string f, part; while (ss >> part && part != "moves") f += part + " ";
        ok = np.setFen(f);
        if (part == "moves") tok = "moves";
        if (ok) { pos = np; hist = {pos.key}; }
      }
      if (tok == "startpos" && ok) { pos = np; hist = {pos.key}; }
      if (ok) {
        if (tok != "moves") ss >> tok;
        if (tok == "moves") {
          string mv;
          while (ss >> mv) {
            Move ml[256]; int n = pos.genLegal(ml); Move f = 0;
            for (int i = 0; i < n; i++) if (mvStr(ml[i]) == mv) f = ml[i];
            if (!f) break;
            pos.make(f); hist.push_back(pos.key);
          }
        }
      }
    } else if (cmd == "go") {
      join();
      long long wt = 0, bt = 0, wi = 0, bi = 0, mt = 0, nodes = 0; int depth = 64, mtg = 30; bool inf = false;
      string t;
      while (ss >> t) {
        if (t == "wtime") ss >> wt; else if (t == "btime") ss >> bt; else if (t == "winc") ss >> wi;
        else if (t == "binc") ss >> bi; else if (t == "movetime") ss >> mt; else if (t == "depth") ss >> depth;
        else if (t == "nodes") ss >> nodes; else if (t == "movestogo") ss >> mtg; else if (t == "infinite") inf = true;
      }
      Limits L; L.depth = depth; L.nodes = nodes;
      if (mt > 0) { L.hardMs = (int)max(5LL, mt - 15); L.softMs = (int)(L.hardMs * 0.6); }
      else if (!inf && (wt || bt)) {
        long long my = pos.side == WHITE ? wt : bt, inc = pos.side == WHITE ? wi : bi;
        long long alloc = my / max(1, mtg) + inc * 3 / 4;
        L.softMs = (int)max(10LL, alloc);
        L.hardMs = (int)max(15LL, min(alloc * 3, max(10LL, my / 2)));
      }
      stopFlag = false;
      Pos p = pos; vector<U64> h = hist;
      worker = thread([&S, p, h, L]() {
        Move m = S->think(p, h, L, nullptr, true);
        printf("bestmove %s\n", m ? mvStr(m).c_str() : "0000"); fflush(stdout);
      });
    } else if (cmd == "stop") { stopFlag = true; join(); }
    else if (cmd == "quit") { stopFlag = true; join(); break; }
    fflush(stdout);
  }
  stopFlag = true; join(); delete S;
}

// ============================================================================
//  เล่นในเทอร์มินัล
// ============================================================================
static void printBoard(const Pos& p, bool flip, Move last) {
  printf("\n");
  for (int ri = 0; ri < 8; ri++) {
    int r = flip ? ri : 7 - ri;
    printf(" %d ", r + 1);
    for (int fi = 0; fi < 8; fi++) {
      int f = flip ? 7 - fi : fi, s = r * 8 + f, pc = p.b[s];
      char c = pc ? " pnbrqk"[typeOf(pc)] : '.';
      if (pc && colorOf(pc) == WHITE) c = (char)toupper(c);
      bool hl = last && ((last & 63) == s || ((last >> 6) & 63) == s);
      printf(hl ? "[%c]" : " %c ", c);
    }
    printf("\n");
  }
  printf("   ");
  for (int fi = 0; fi < 8; fi++) printf(" %c ", 'a' + (flip ? 7 - fi : fi));
  printf("\n   (ตัวใหญ่ = ขาว, ตัวเล็ก = ดำ)\n");
}

static string gameOverMsg(const Pos& p, const vector<U64>& hist) {
  if (p.legalCount() == 0) return p.inCheck() ? (p.side == WHITE ? "รุกจน! ดำชนะ" : "รุกจน! ขาวชนะ") : "เสมอ (สเตลเมต)";
  if (insufficient(p)) return "เสมอ (หมากไม่พอรุกจน)";
  if (p.half >= 100) return "เสมอ (กฎ 50 ตา)";
  int c = 0; for (U64 k : hist) if (k == p.key) c++;
  if (c >= 3) return "เสมอ (ซ้ำตำแหน่ง 3 ครั้ง)";
  return "";
}

static void playMode(Brain& brain) {
  Searcher S(&brain, 20);
  printf("=== UnChess : Unknown Variable Engine ===\n");
  printf("สมองที่ใช้: %d ตัวแปรที่ค้นพบเอง\n", (int)brain.gs.size());
  printf("พิมพ์ตาเดินแบบ e2e4 หรือ Nf3 | คำสั่ง: help\n");
  string line;
  int sec = 3;
  while (true) {
    printf("\nเล่นสีอะไร? (w=ขาว / b=ดำ / q=ออก): "); fflush(stdout);
    if (!getline(cin, line) || line == "q") return;
    int human = (line == "b" || line == "B") ? BLACK : WHITE;
    vector<Pos> stack; vector<U64> hist; Pos pos; pos.setFen(STARTFEN);
    stack.push_back(pos); hist.push_back(pos.key); S.clearTT();
    Move last = 0;
    while (true) {
      printBoard(pos, human == BLACK, last);
      string over = gameOverMsg(pos, hist);
      if (!over.empty()) { printf("\n*** %s ***\n", over.c_str()); break; }
      if (pos.inCheck()) printf("** รุก! **\n");
      if (pos.side != human) {
        printf("AI กำลังคิด (%d วินาที)...\n", sec);
        Limits L; L.hardMs = sec * 1000; L.softMs = sec * 600; int sc = 0;
        Move m = S.think(pos, hist, L, &sc, false);
        if (!m) break;
        printf("AI เดิน: %s   (ประเมิน %+.2f ในมุมมอง AI, ลึก %d)\n", toSAN(pos, m).c_str(), sc / 100.0, S.completed);
        pos.make(m); hist.push_back(pos.key); stack.push_back(pos); last = m;
        continue;
      }
      printf("ตาคุณ > "); fflush(stdout);
      if (!getline(cin, line)) return;
      if (line == "quit" || line == "q") return;
      if (line == "help") { printf("undo = ย้อนตา | new = เริ่มใหม่ | hint = ขอคำใบ้ | t N = ให้ AI คิด N วินาที | eval | fen | brain | quit\n"); continue; }
      if (line == "new") break;
      if (line == "fen") { printf("%s\n", pos.fen().c_str()); continue; }
      if (line == "eval") { printf("ประเมินตำแหน่ง (ขาวนำ = บวก): %+.2f\n", brain.evalWhite(pos) / 100.0); continue; }
      if (line == "brain") { for (size_t j = 0; j < brain.gs.size(); j++) printf(" %zu) %s [%+.1f/%+.1f]\n", j + 1, genomeDesc(brain.gs[j]).c_str(), brain.gs[j].wmg, brain.gs[j].weg); if (brain.gs.empty()) printf("ยังไม่มีตัวแปรที่ค้นพบ (รัน ./unchess learn)\n"); continue; }
      if (line.rfind("t ", 0) == 0) { sec = max(1, min(120, atoi(line.c_str() + 2))); printf("ตั้งเวลาคิด %d วินาที\n", sec); continue; }
      if (line == "undo") {
        if (stack.size() >= 3) { stack.pop_back(); stack.pop_back(); hist.pop_back(); hist.pop_back(); pos = stack.back(); last = 0; }
        else printf("ย้อนไม่ได้\n");
        continue;
      }
      if (line == "hint") {
        Limits L; L.hardMs = sec * 1000; L.softMs = sec * 600;
        Move m = S.think(pos, hist, L, nullptr, false);
        printf("คำใบ้: %s\n", toSAN(pos, m).c_str()); continue;
      }
      Move m = parseMove(pos, line);
      if (!m) {
        printf("ตาเดินนี้ไม่ถูกกฎ/อ่านไม่ออก. ตาที่เดินได้: ");
        Move ml[256]; int n = pos.genLegal(ml);
        for (int i = 0; i < n; i++) printf("%s ", toSAN(pos, ml[i]).c_str());
        printf("\n"); continue;
      }
      pos.make(m); hist.push_back(pos.key); stack.push_back(pos); last = m;
    }
  }
}

// ============================================================================
//  ทดสอบความถูกต้อง
// ============================================================================
static int selfTest() {
  int fails = 0;
  struct T { const char* fen; int d; uint64_t n; };
  T tests[] = {
    {STARTFEN, 5, 4865609ULL},
    {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603ULL},
    {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624ULL},
    {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333ULL},
    {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487ULL},
    {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594ULL},
  };
  printf("[1] Perft (ตรวจกฎทั้งหมด: การเดิน, กิน, ผ่าน, เลื่อนขั้น, ปราสาท)\n");
  for (auto& t : tests) {
    Pos p; if (!p.setFen(t.fen)) { printf("  FEN เสีย\n"); fails++; continue; }
    auto st = chrono::steady_clock::now();
    uint64_t n = perft(p, t.d);
    double ms = chrono::duration<double, milli>(chrono::steady_clock::now() - st).count();
    printf("  depth %d = %llu  คาดว่า %llu  %s  (%.0f ms)\n", t.d, (unsigned long long)n, (unsigned long long)t.n, n == t.n ? "ผ่าน" : "*** ไม่ผ่าน ***", ms);
    if (n != t.n) fails++;
  }
  printf("[2] Fuzz: เกมสุ่ม 300 เกม ตรวจ hash/FEN/SAN/ความถูกต้องทุกตา\n");
  mt19937 rng(12345); long long plies = 0;
  for (int g = 0; g < 300; g++) {
    Pos p; p.setFen(STARTFEN);
    for (int i = 0; i < 200; i++) {
      Move ml[256]; int n = p.genLegal(ml);
      if (n == 0 || insufficient(p)) break;
      if (p.key != p.computeKey()) { printf("  hash ผิด\n"); fails++; break; }
      Pos q; if (!q.setFen(p.fen()) || q.key != p.key) { printf("  FEN roundtrip ผิด: %s\n", p.fen().c_str()); fails++; break; }
      for (int k = 0; k < n; k++) {
        Pos nx = p; nx.make(ml[k]);
        if (nx.attacked(nx.ksq[p.side], p.side ^ 1)) { printf("  ตาผิดกฎหลุดมา\n"); fails++; }
        if (nx.key != nx.computeKey()) { printf("  hash ตาเดินผิด\n"); fails++; }
        if (parseMove(p, toSAN(p, ml[k])) != ml[k]) { printf("  SAN roundtrip ผิด %s %s\n", p.fen().c_str(), toSAN(p, ml[k]).c_str()); fails++; }
        if (parseMove(p, mvStr(ml[k])) != ml[k]) { printf("  coord roundtrip ผิด\n"); fails++; }
      }
      p.make(ml[rng() % n]); plies++;
    }
  }
  printf("  ตรวจแล้ว %lld ตา -> %s\n", plies, fails ? "มีปัญหา" : "ผ่านทั้งหมด");
  printf("[3] AI เล่นกันเอง 6 เกม (ลึก 3) ตรวจว่าทุกตาถูกกฎและเกมจบถูกต้อง\n");
  Brain br; Searcher S(&br, 16); Limits L; L.depth = 3;
  for (int g = 0; g < 6; g++) {
    Pos p; p.setFen(STARTFEN); vector<U64> h = {p.key}; S.clearTT(); int ply = 0;
    while (ply < 400) {
      Move ml[256]; int n = p.genLegal(ml);
      if (n == 0 || p.half >= 100 || insufficient(p)) break;
      int c = 0; for (U64 k : h) if (k == p.key) c++;
      if (c >= 3) break;
      Move m = S.think(p, h, L, nullptr, false);
      bool ok = false; for (int i = 0; i < n; i++) if (ml[i] == m) ok = true;
      if (!ok) { printf("  *** AI เลือกตาผิดกฎ! ***\n"); fails++; break; }
      p.make(m); h.push_back(p.key); ply++;
    }
    printf("  เกม %d: %d ตา, จบด้วย %s\n", g + 1, ply, p.legalCount() == 0 ? (p.inCheck() ? "รุกจน" : "สเตลเมต") : "เสมอ/จำกัดตา");
  }
  printf("\nผลรวม: %s\n", fails ? "มีบางอย่างไม่ผ่าน" : "ผ่านทั้งหมด ✔");
  return fails ? 1 : 0;
}

int main(int argc, char** argv) {
  initTables();
  ios::sync_with_stdio(true);
  string brainFile = "unchess.brain";
  string mode = argc > 1 ? argv[1] : "play";
  if (mode == "perft") {
    int d = argc > 2 ? atoi(argv[2]) : 5; string f = STARTFEN;
    if (argc > 3) { f = ""; for (int i = 3; i < argc; i++) f += string(argv[i]) + " "; }
    Pos p; if (!p.setFen(f)) { printf("FEN ผิด\n"); return 1; }
    Move ml[256]; int n = p.genLegal(ml); uint64_t tot = 0;
    for (int i = 0; i < n; i++) { Pos nx = p; nx.make(ml[i]); uint64_t c = d > 1 ? perft(nx, d - 1) : 1; printf("%s: %llu\n", mvStr(ml[i]).c_str(), (unsigned long long)c); tot += c; }
    printf("\nรวม: %llu\n", (unsigned long long)tot); return 0;
  }
  if (mode == "test") return selfTest();
  Brain brain; brain.load(brainFile);
  if (mode == "uci") { uciLoop(brain, brainFile); return 0; }
  if (mode == "learn") { learnMode(argc, argv, brainFile); return 0; }
  if (mode == "play") { if (argc > 2) brain.load(argv[2]); playMode(brain); return 0; }
  printf("วิธีใช้: ./unchess [play|uci|learn|test|perft N [fen]]\n");
  return 0;
}
