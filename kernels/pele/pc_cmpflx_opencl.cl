#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "dodecane_lu_opencl.h"

#define URHO 0
#define UMX 1
#define UMY 2
#define UMZ 3
#define UEDEN 4
#define UEINT 5
#define UTEMP 6
#define UFA 7

#define QRHO 0
#define QU 1
#define QV 2
#define QW 3
#define QGAME 4
#define QPRES 5
#define QREINT 6
#define QTEMP 7
#define QFA 8

#define QGAMC 0
#define QC 1
#define QCSML 2
#define QDPDR 3
#define QDPDE 4
#define QRSPEC 5

#define GDRHO 0
#define GDU 1
#define GDV 2
#define GDW 3
#define GDPRES 4
#define GDGAME 5

#define NUM_ADV 0
#define NUM_AUX 0
#define NUM_LIN 0

#define UFS (UFA + NUM_ADV)
#define QFS (QFA + NUM_ADV)

static __constant double constants_RU = 8.31446261815324e7;
static __constant double cl_dbl_epsilon = 2.2204460492503131e-16;
static __constant double cl_dbl_min = 2.2250738585072014e-308;

static inline double constants_smallu(void)
{
  return 1.0e-12;
}

static inline double constants_small_num(void)
{
  return 1.0e-8;
}

static inline double constants_very_small_num(void)
{
  return cl_dbl_epsilon * 1e-100;
}

static inline void
pc_cmpflx_passive(const double ustar, const double flxrho, const double ql,
                  const double qr, __global double *flx)
{
  if (ustar > 0.0) {
    *flx = flxrho * ql;
  } else if (ustar < 0.0) {
    *flx = flxrho * qr;
  } else {
    *flx = flxrho * 0.5 * (ql + qr);
  }
}

static inline void RPY2Cs(const double R, const double P,
                          const double Y[NUM_SPECIES], double *Cs)
{
  double tmp[NUM_SPECIES];
  double wbar = 0.0;
  double T;
  double Cv = 0.0;
  double G;

  CKMMWY(Y, &wbar);
  T = P * wbar / (R * constants_RU);
  CKCVMS(T, tmp);
  for (int i = 0; i < NUM_SPECIES; i++) {
    Cv += Y[i] * tmp[i];
  }
  G = (wbar * Cv + constants_RU) / (wbar * Cv);
  *Cs = sqrt(G * P / R);
}

static inline void RYP2E(const double R, const double Y[NUM_SPECIES],
                         const double P, double *E)
{
  double wbar = 0.0;
  double T;
  double ei[NUM_SPECIES];

  CKMMWY(Y, &wbar);
  T = P * wbar / (R * constants_RU);
  CKUMS(T, ei);
  *E = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    *E += Y[n] * ei[n];
  }
}

static inline void riemann(
    const double rl, const double ul, const double vl, const double v2l,
    const double pl, const double spl[NUM_SPECIES], const double rr,
    const double ur, const double vr, const double v2r, const double pr,
    const double spr[NUM_SPECIES], const int bc_test_val, const double cav,
    __private double *ustar, __global double *uflx_rho,
    __private double uflx_rhoY[NUM_SPECIES], __global double *uflx_u,
    __global double *uflx_v, __global double *uflx_w, __global double *uflx_eden,
    __global double *uflx_eint, __global double *qint_iu,
    __global double *qint_iv1, __global double *qint_iv2,
    __global double *qint_gdpres, __global double *qint_gdgame)
{
  const double wsmall = cl_dbl_min;

  double gdnv_state_massfrac[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = spl[n];
  }
  double cl = 0.0;
  RPY2Cs(rl, pl, gdnv_state_massfrac, &cl);

  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = spr[n];
  }
  double cr = 0.0;
  RPY2Cs(rr, pr, gdnv_state_massfrac, &cr);

  const double wl = fmax(wsmall, cl * rl);
  const double wr = fmax(wsmall, cr * rr);
  const double pstar = fmax(
      cl_dbl_min, ((wr * pl + wl * pr) + wl * wr * (ul - ur)) / (wl + wr));
  *ustar = ((wl * ul + wr * ur) + (pl - pr)) / (wl + wr);

  int mask = *ustar > 0.0;
  double ro = 0.0;
  double rspo[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    rspo[n] = mask ? rl * spl[n] : rr * spr[n];
    ro += rspo[n];
  }
  double uo = mask ? ul : ur;
  double po = mask ? pl : pr;

  mask = fabs(*ustar) < constants_smallu() * 0.5 * (fabs(ul) + fabs(ur)) ||
         *ustar == 0.0;
  *ustar = mask ? 0.0 : *ustar;
  ro = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    rspo[n] = mask ? 0.5 * (rl * spl[n] + rr * spr[n]) : rspo[n];
    ro += rspo[n];
  }
  uo = mask ? 0.5 * (ul + ur) : uo;
  po = mask ? 0.5 * (pl + pr) : po;

  double gdnv_state_rho = ro;
  double gdnv_state_p = po;
  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = rspo[n] / ro;
  }
  double gdnv_state_e;
  RYP2E(gdnv_state_rho, gdnv_state_massfrac, gdnv_state_p, &gdnv_state_e);
  double co;
  RPY2Cs(gdnv_state_rho, gdnv_state_p, gdnv_state_massfrac, &co);

  const double drho = (pstar - po) / (co * co);
  double rstar = 0.0;
  double rspstar[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    const double spon = rspo[n] / ro;
    rspstar[n] = fmax(0.0, rspo[n] + drho * spon);
    rstar += rspstar[n];
  }
  gdnv_state_rho = rstar;
  gdnv_state_p = pstar;
  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = rspstar[n] / rstar;
  }
  RYP2E(gdnv_state_rho, gdnv_state_massfrac, gdnv_state_p, &gdnv_state_e);
  double cstar;
  RPY2Cs(gdnv_state_rho, gdnv_state_p, gdnv_state_massfrac, &cstar);

  const double sgnm = copysign(1.0, *ustar);

  double spout = co - sgnm * uo;
  double spin = cstar - sgnm * *ustar;
  const double ushock = 0.5 * (spin + spout);

  mask = pstar < po;
  spout = mask ? spout : ushock;
  spin = mask ? spin : ushock;

  const double scr = (fabs(spout - spin) < constants_very_small_num())
                         ? constants_small_num() * cav
                         : spout - spin;
  const double frac = fmax(0.0, fmin(1.0, (1.0 + (spout + spin) / scr) * 0.5));

  mask = *ustar > 0.0;
  *qint_iv1 = mask ? vl : vr;
  *qint_iv2 = mask ? v2l : v2r;

  mask = (*ustar == 0.0);
  *qint_iv1 = mask ? 0.5 * (vl + vr) : *qint_iv1;
  *qint_iv2 = mask ? 0.5 * (v2l + v2r) : *qint_iv2;
  double rgd = 0.0;
  double rspgd[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    rspgd[n] = frac * rspstar[n] + (1.0 - frac) * rspo[n];
    rgd += rspgd[n];
  }
  *qint_iu = frac * *ustar + (1.0 - frac) * uo;
  *qint_gdpres = frac * pstar + (1.0 - frac) * po;
  gdnv_state_rho = rgd;
  gdnv_state_p = *qint_gdpres;
  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = rspgd[n] / rgd;
  }
  RYP2E(gdnv_state_rho, gdnv_state_massfrac, gdnv_state_p, &gdnv_state_e);

  mask = (spout < 0.0);
  rgd = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    rspgd[n] = mask ? rspo[n] : rspgd[n];
    rgd += rspgd[n];
  }
  *qint_iu = mask ? uo : *qint_iu;
  *qint_gdpres = mask ? po : *qint_gdpres;

  mask = (spin >= 0.0);
  rgd = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    rspgd[n] = mask ? rspstar[n] : rspgd[n];
    rgd += rspgd[n];
  }
  *qint_iu = mask ? *ustar : *qint_iu;
  *qint_gdpres = mask ? pstar : *qint_gdpres;

  gdnv_state_rho = rgd;
  gdnv_state_p = *qint_gdpres;
  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = rspgd[n] / rgd;
  }
  RYP2E(gdnv_state_rho, gdnv_state_massfrac, gdnv_state_p, &gdnv_state_e);
  const double regd = gdnv_state_rho * gdnv_state_e;

  *qint_gdgame = *qint_gdpres / regd + 1.0;
  *qint_iu = bc_test_val * *qint_iu;
  *uflx_rho = rgd * *qint_iu;
  for (int n = 0; n < NUM_SPECIES; n++) {
    uflx_rhoY[n] = rspgd[n] * *qint_iu;
  }
  *uflx_u = *uflx_rho * *qint_iu + *qint_gdpres;
  *uflx_v = *uflx_rho * *qint_iv1;
  *uflx_w = *uflx_rho * *qint_iv2;
  const double rhoetot =
      regd + 0.5 * rgd *
                 (*qint_iu * *qint_iu + *qint_iv1 * *qint_iv1 +
                  *qint_iv2 * *qint_iv2);
  *uflx_eden = *qint_iu * (rhoetot + *qint_gdpres);
  *uflx_eint = *qint_iu * regd;
}

static inline void pc_cmpflx(
    const int i, const int j, const int k, const int bclo, const int bchi,
    const int domlo, const int domhi, __global const double *ql,
    const int ql_jstride, const int ql_kstride, const int ql_nstride,
    const int ql_beginx, const int ql_beginy, const int ql_beginz,
    __global const double *qr, const int qr_jstride, const int qr_kstride,
    const int qr_nstride, const int qr_beginx, const int qr_beginy,
    const int qr_beginz, __global double *flx, const int flx_jstride,
    const int flx_kstride, const int flx_nstride, const int flx_beginx,
    const int flx_beginy, const int flx_beginz, __global double *q,
    const int q_jstride, const int q_kstride, const int q_nstride,
    const int q_beginx, const int q_beginy, const int q_beginz,
    __global const double *qa, const int qa_jstride, const int qa_kstride,
    const int qa_nstride, const int qa_beginx, const int qa_beginy,
    const int qa_beginz, const int dir)
{
  double cav;
  double ustar;
  double spl[NUM_SPECIES];
  double spr[NUM_SPECIES];
  int idx;
  int IU;
  int IV;
  int IV2;
  int GU;
  int GV;
  int GV2;
  int f_idx[3];

  if (dir == 0) {
    IU = QU;
    IV = QV;
    IV2 = QW;
    GU = GDU;
    GV = GDV;
    GV2 = GDW;
    cav = 0.5 *
          (qa[(i - qa_beginx) + (j - qa_beginy) * qa_jstride +
              (k - qa_beginz) * qa_kstride + QC * qa_nstride] +
           qa[(i - 1 - qa_beginx) + (j - qa_beginy) * qa_jstride +
              (k - qa_beginz) * qa_kstride + QC * qa_nstride]);
    f_idx[0] = UMX;
    f_idx[1] = UMY;
    f_idx[2] = UMZ;
  } else if (dir == 1) {
    IU = QV;
    IV = QU;
    IV2 = QW;
    GU = GDV;
    GV = GDU;
    GV2 = GDW;
    cav = 0.5 *
          (qa[(i - qa_beginx) + (j - qa_beginy) * qa_jstride +
              (k - qa_beginz) * qa_kstride + QC * qa_nstride] +
           qa[(i - qa_beginx) + (j - 1 - qa_beginy) * qa_jstride +
              (k - qa_beginz) * qa_kstride + QC * qa_nstride]);
    f_idx[0] = UMY;
    f_idx[1] = UMX;
    f_idx[2] = UMZ;
  } else {
    IU = QW;
    IV = QU;
    IV2 = QV;
    GU = GDW;
    GV = GDU;
    GV2 = GDV;
    cav = 0.5 *
          (qa[(i - qa_beginx) + (j - qa_beginy) * qa_jstride +
              (k - qa_beginz) * qa_kstride + QC * qa_nstride] +
           qa[(i - qa_beginx) + (j - qa_beginy) * qa_jstride +
              (k - 1 - qa_beginz) * qa_kstride + QC * qa_nstride]);
    f_idx[0] = UMZ;
    f_idx[1] = UMX;
    f_idx[2] = UMY;
  }

  for (int sp = 0; sp < NUM_SPECIES; ++sp) {
    spl[sp] = ql[(i - ql_beginx) + (j - ql_beginy) * ql_jstride +
                (k - ql_beginz) * ql_kstride + (QFS + sp) * ql_nstride];
    spr[sp] = qr[(i - qr_beginx) + (j - qr_beginy) * qr_jstride +
                (k - qr_beginz) * qr_kstride + (QFS + sp) * qr_nstride];
  }

  const double ul =
      ql[(i - ql_beginx) + (j - ql_beginy) * ql_jstride +
         (k - ql_beginz) * ql_kstride + IU * ql_nstride];
  const double vl =
      ql[(i - ql_beginx) + (j - ql_beginy) * ql_jstride +
         (k - ql_beginz) * ql_kstride + IV * ql_nstride];
  const double v2l =
      ql[(i - ql_beginx) + (j - ql_beginy) * ql_jstride +
         (k - ql_beginz) * ql_kstride + IV2 * ql_nstride];
  const double pl =
      ql[(i - ql_beginx) + (j - ql_beginy) * ql_jstride +
         (k - ql_beginz) * ql_kstride + QPRES * ql_nstride];
  const double rhol =
      ql[(i - ql_beginx) + (j - ql_beginy) * ql_jstride +
         (k - ql_beginz) * ql_kstride + QRHO * ql_nstride];

  const double ur =
      qr[(i - qr_beginx) + (j - qr_beginy) * qr_jstride +
         (k - qr_beginz) * qr_kstride + IU * qr_nstride];
  const double vr =
      qr[(i - qr_beginx) + (j - qr_beginy) * qr_jstride +
         (k - qr_beginz) * qr_kstride + IV * qr_nstride];
  const double v2r =
      qr[(i - qr_beginx) + (j - qr_beginy) * qr_jstride +
         (k - qr_beginz) * qr_kstride + IV2 * qr_nstride];
  const double pr =
      qr[(i - qr_beginx) + (j - qr_beginy) * qr_jstride +
         (k - qr_beginz) * qr_kstride + QPRES * qr_nstride];
  const double rhor =
      qr[(i - qr_beginx) + (j - qr_beginy) * qr_jstride +
         (k - qr_beginz) * qr_kstride + QRHO * qr_nstride];

  if (dir == 2) {
    idx = k;
  } else {
    idx = (dir == 0) ? i : j;
  }

  (void)idx;
  (void)bclo;
  (void)bchi;
  (void)domlo;
  (void)domhi;

  const int bc_test_val = 1;
  double dummy_flx[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    dummy_flx[n] = 0.0;
  }

  riemann(rhol, ul, vl, v2l, pl, spl, rhor, ur, vr, v2r, pr, spr, bc_test_val,
          cav, &ustar,
          &flx[(i - flx_beginx) + (j - flx_beginy) * flx_jstride +
               (k - flx_beginz) * flx_kstride + URHO * flx_nstride],
          dummy_flx,
          &flx[(i - flx_beginx) + (j - flx_beginy) * flx_jstride +
               (k - flx_beginz) * flx_kstride + f_idx[0] * flx_nstride],
          &flx[(i - flx_beginx) + (j - flx_beginy) * flx_jstride +
               (k - flx_beginz) * flx_kstride + f_idx[1] * flx_nstride],
          &flx[(i - flx_beginx) + (j - flx_beginy) * flx_jstride +
               (k - flx_beginz) * flx_kstride + f_idx[2] * flx_nstride],
          &flx[(i - flx_beginx) + (j - flx_beginy) * flx_jstride +
               (k - flx_beginz) * flx_kstride + UEDEN * flx_nstride],
          &flx[(i - flx_beginx) + (j - flx_beginy) * flx_jstride +
               (k - flx_beginz) * flx_kstride + UEINT * flx_nstride],
          &q[(i - q_beginx) + (j - q_beginy) * q_jstride +
             (k - q_beginz) * q_kstride + GU * q_nstride],
          &q[(i - q_beginx) + (j - q_beginy) * q_jstride +
             (k - q_beginz) * q_kstride + GV * q_nstride],
          &q[(i - q_beginx) + (j - q_beginy) * q_jstride +
             (k - q_beginz) * q_kstride + GV2 * q_nstride],
          &q[(i - q_beginx) + (j - q_beginy) * q_jstride +
             (k - q_beginz) * q_kstride + GDPRES * q_nstride],
          &q[(i - q_beginx) + (j - q_beginy) * q_jstride +
             (k - q_beginz) * q_kstride + GDGAME * q_nstride]);

  const double flxrho =
      flx[(i - flx_beginx) + (j - flx_beginy) * flx_jstride +
          (k - flx_beginz) * flx_kstride + URHO * flx_nstride];

  for (int n = 0; n < NUM_SPECIES; n++) {
    const int qc = QFS + n;
    pc_cmpflx_passive(
        ustar, flxrho,
        ql[(i - ql_beginx) + (j - ql_beginy) * ql_jstride +
           (k - ql_beginz) * ql_kstride + qc * ql_nstride],
        qr[(i - qr_beginx) + (j - qr_beginy) * qr_jstride +
           (k - qr_beginz) * qr_kstride + qc * qr_nstride],
        &flx[(i - flx_beginx) + (j - flx_beginy) * flx_jstride +
             (k - flx_beginz) * flx_kstride + (UFS + n) * flx_nstride]);
  }
}

__kernel void pc_cmpflx_launch(
    const int bclo, const int bchi, const int domlo, const int domhi,
    const int ncells, const int lenx, const int lenxy, const int lox,
    const int loy, const int loz, __global const double *qlxy,
    const int qlxy_jstride, const int qlxy_kstride, const int qlxy_nstride,
    const int qlxy_beginx, const int qlxy_beginy, const int qlxy_beginz,
    __global const double *qrxy, const int qrxy_jstride, const int qrxy_kstride,
    const int qrxy_nstride, const int qrxy_beginx, const int qrxy_beginy,
    const int qrxy_beginz, __global double *flxy, const int flxy_jstride,
    const int flxy_kstride, const int flxy_nstride, const int flxy_beginx,
    const int flxy_beginy, const int flxy_beginz, __global double *qxy,
    const int qxy_jstride, const int qxy_kstride, const int qxy_nstride,
    const int qxy_beginx, const int qxy_beginy, const int qxy_beginz,
    __global const double *qlxz, const int qlxz_jstride, const int qlxz_kstride,
    const int qlxz_nstride, const int qlxz_beginx, const int qlxz_beginy,
    const int qlxz_beginz, __global const double *qrxz,
    const int qrxz_jstride, const int qrxz_kstride, const int qrxz_nstride,
    const int qrxz_beginx, const int qrxz_beginy, const int qrxz_beginz,
    __global double *flxz, const int flxz_jstride, const int flxz_kstride,
    const int flxz_nstride, const int flxz_beginx, const int flxz_beginy,
    const int flxz_beginz, __global double *qxz, const int qxz_jstride,
    const int qxz_kstride, const int qxz_nstride, const int qxz_beginx,
    const int qxz_beginy, const int qxz_beginz, __global const double *qaux,
    const int qaux_jstride, const int qaux_kstride, const int qaux_nstride,
    const int qaux_beginx, const int qaux_beginy, const int qaux_beginz,
    const int dir)
{
  for (int icell = (int)(get_global_id(0)); icell < ncells;
       icell += (int)get_global_size(0)) {
    int k = icell / lenxy;
    int j = (icell - k * lenxy) / lenx;
    int i = (icell - k * lenxy) - j * lenx;
    i += lox;
    j += loy;
    k += loz;

    pc_cmpflx(i, j, k, bclo, bchi, domlo, domhi, qlxy, qlxy_jstride,
              qlxy_kstride, qlxy_nstride, qlxy_beginx, qlxy_beginy,
              qlxy_beginz, qrxy, qrxy_jstride, qrxy_kstride, qrxy_nstride,
              qrxy_beginx, qrxy_beginy, qrxy_beginz, flxy, flxy_jstride,
              flxy_kstride, flxy_nstride, flxy_beginx, flxy_beginy,
              flxy_beginz, qxy, qxy_jstride, qxy_kstride, qxy_nstride,
              qxy_beginx, qxy_beginy, qxy_beginz, qaux, qaux_jstride,
              qaux_kstride, qaux_nstride, qaux_beginx, qaux_beginy,
              qaux_beginz, dir);

    pc_cmpflx(i, j, k, bclo, bchi, domlo, domhi, qlxz, qlxz_jstride,
              qlxz_kstride, qlxz_nstride, qlxz_beginx, qlxz_beginy,
              qlxz_beginz, qrxz, qrxz_jstride, qrxz_kstride, qrxz_nstride,
              qrxz_beginx, qrxz_beginy, qrxz_beginz, flxz, flxz_jstride,
              flxz_kstride, flxz_nstride, flxz_beginx, flxz_beginy,
              flxz_beginz, qxz, qxz_jstride, qxz_kstride, qxz_nstride,
              qxz_beginx, qxz_beginy, qxz_beginz, qaux, qaux_jstride,
              qaux_kstride, qaux_nstride, qaux_beginx, qaux_beginy,
              qaux_beginz, dir);
  }
}
