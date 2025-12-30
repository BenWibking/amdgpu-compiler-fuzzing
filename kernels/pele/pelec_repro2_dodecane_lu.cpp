/**********************************************************************************************/
/* Run via:                                                                                   */
/* pelec_repro_dodecane_lu POOL_SIZE NUM_TRIALS CFRHS_MULTI_KERNEL CFRHS_MIN_BLOCKS RTOL ATOL */
/*   POOL_SIZE: value in GB                                                                   */
/*   NUM_TRIALS: integer > 0 ...                                                              */
/*   CFRHS_MULTI_KERNEL: 0 (call the single kernel monolithic version) or                     */
/*                       1 (call the version where it is split into 3 subkernels)             */
/*   CRHS_MIN_BLOCKS: used in launch_bounds to force concurrency. valid values are 1,2,3,4    */
/*   RTOL: relative error tolerance between this impl and a full Pele run (optional)          */
/*   ATOL: absolute error tolerance between this impl and a full Pele run (optional)          */
/**********************************************************************************************/

#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#define WARP_SIZE 64
#define CHECK_BAD

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

#define AMREX_GPU_DEVICE __device__
#define AMREX_GPU_HOST_DEVICE __device__
#define AMREX_FORCE_INLINE __forceinline__
#define AMREX_NO_INLINE  __attribute__((noinline))

#define NAME "dodecane_lu"
#include "dodecane_lu.h"
#define DEBUG

#define HIP_CALL(call)                                   \
	do {                                                  \
	hipError_t err = call;                                \
	if (hipSuccess != err) {                              \
	printf("HIP ERROR (code = %d, %s) at %s:%d\n", err,   \
			 hipGetErrorString(err), __FILE__, __LINE__);   \
	assert(0);                                            \
	exit(1);                                              \
	}                                                     \
} while (0)

struct Constants
{
  static constexpr double gamma = 1.4;
  static constexpr double RU = 8.31446261815324e7;
  static constexpr double RUC = 1.98721558317399615845;
  static constexpr double PATM = 1.01325e+06;
  static constexpr double AIRMW = 28.97;
  static constexpr double Avna = 6.022140857e23;
};

namespace constants {
AMREX_GPU_HOST_DEVICE constexpr double
smallu()
{
  return 1.0e-12;
}
AMREX_GPU_HOST_DEVICE constexpr double
small_num()
{
  return 1.0e-8;
}
AMREX_GPU_HOST_DEVICE constexpr double
very_small_num()
{
  return std::numeric_limits<double>::epsilon() * 1e-100;
}
} // namespace constants

/****************************************************************/
/* Interface routines                                           */
/****************************************************************/

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pc_cmpflx_passive(
  const double ustar,
  const double flxrho,
  const double& ql,
  const double& qr,
  double& flx)
{
  flx = (ustar > 0.0)   ? flxrho * ql
        : (ustar < 0.0) ? flxrho * qr
                        : flxrho * 0.5 * (ql + qr);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
static void RPY2Cs(const double R,
		   const double P,
		   const double Y[NUM_SPECIES],
		   double& Cs)
{
  double tmp[NUM_SPECIES];
  double wbar = 0.0;
  CKMMWY(Y, wbar);
  double T = P * wbar / (R * Constants::RU);
  CKCVMS(T, tmp);
  double Cv = 0.0;
  for (int i = 0; i < NUM_SPECIES; i++) {
    Cv += Y[i] * tmp[i];
  }
  double G = (wbar * Cv + Constants::RU) / (wbar * Cv);
  Cs = std::sqrt(G * P / R);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
static void RYP2E(const double R,
		  const double Y[NUM_SPECIES],
		  const double P,
		  double& E)
{
  double wbar = 0.0;
  CKMMWY(Y, wbar);
  double T = P * wbar / (R * Constants::RU);
  double ei[NUM_SPECIES];
  CKUMS(T, ei);
  E = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    E += Y[n] * ei[n];
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
riemann(
  const double rl,
  const double ul,
  const double vl,
  const double v2l,
  const double pl,
  const double spl[NUM_SPECIES],
  const double rr,
  const double ur,
  const double vr,
  const double v2r,
  const double pr,
  const double spr[NUM_SPECIES],
  const int bc_test_val,
  const double cav,
  double& ustar,
  double& uflx_rho,
  double uflx_rhoY[NUM_SPECIES],
  double& uflx_u,
  double& uflx_v,
  double& uflx_w,
  double& uflx_eden,
  double& uflx_eint,
  double& qint_iu,
  double& qint_iv1,
  double& qint_iv2,
  double& qint_gdpres,
  double& qint_gdgame)
{
  const double wsmall = std::numeric_limits<double>::min();

  double gdnv_state_massfrac[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = spl[n];
  }
  double cl = 0.0;
  RPY2Cs(rl, pl, gdnv_state_massfrac, cl);

  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = spr[n];
  }
  double cr = 0.0;
  RPY2Cs(rr, pr, gdnv_state_massfrac, cr);

  const double wl = std::max(wsmall, cl * rl);
  const double wr = std::max(wsmall, cr * rr);
  const double pstar = std::max(
    std::numeric_limits<double>::min(),
    ((wr * pl + wl * pr) + wl * wr * (ul - ur)) / (wl + wr));
  ustar = ((wl * ul + wr * ur) + (pl - pr)) / (wl + wr);

  bool mask = ustar > 0.0;
  double ro = 0.0;
  double rspo[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    rspo[n] = mask ? rl * spl[n] : rr * spr[n];
    ro += rspo[n];
  }
  double uo = mask ? ul : ur;
  double po = mask ? pl : pr;

  mask = std::abs(ustar) <
           constants::smallu() * 0.5 * (std::abs(ul) + std::abs(ur)) ||
         ustar == 0.0;
  ustar = mask ? 0.0 : ustar;
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
  RYP2E(gdnv_state_rho, gdnv_state_massfrac, gdnv_state_p, gdnv_state_e);
  double co;
  RPY2Cs(gdnv_state_rho, gdnv_state_p, gdnv_state_massfrac, co);

  const double drho = (pstar - po) / (co * co);
  double rstar = 0.0;
  double rspstar[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    const double spon = rspo[n] / ro;
    rspstar[n] = std::max(0.0, rspo[n] + drho * spon);
    rstar += rspstar[n];
  }
  gdnv_state_rho = rstar;
  gdnv_state_p = pstar;
  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = rspstar[n] / rstar;
  }
  RYP2E(gdnv_state_rho, gdnv_state_massfrac, gdnv_state_p, gdnv_state_e);
  double cstar;
  RPY2Cs(gdnv_state_rho, gdnv_state_p, gdnv_state_massfrac, cstar);

  const double sgnm = std::copysign(1.0, ustar);

  double spout = co - sgnm * uo;
  double spin = cstar - sgnm * ustar;
  const double ushock = 0.5 * (spin + spout);

  mask = pstar < po;
  spout = mask ? spout : ushock;
  spin = mask ? spin : ushock;

  const double scr = (std::abs(spout - spin) < constants::very_small_num())
                            ? constants::small_num() * cav
                            : spout - spin;
  const double frac = std::max(
    0.0, std::min(1.0, (1.0 + (spout + spin) / scr) * 0.5));

  mask = ustar > 0.0;
  qint_iv1 = mask ? vl : vr;
  qint_iv2 = mask ? v2l : v2r;

  mask = (ustar == 0.0);
  qint_iv1 = mask ? 0.5 * (vl + vr) : qint_iv1;
  qint_iv2 = mask ? 0.5 * (v2l + v2r) : qint_iv2;
  double rgd = 0.0;
  double rspgd[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    rspgd[n] = frac * rspstar[n] + (1.0 - frac) * rspo[n];
    rgd += rspgd[n];
  }
  qint_iu = frac * ustar + (1.0 - frac) * uo;
  qint_gdpres = frac * pstar + (1.0 - frac) * po;
  gdnv_state_rho = rgd;
  gdnv_state_p = qint_gdpres;
  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = rspgd[n] / rgd;
  }
  RYP2E(gdnv_state_rho, gdnv_state_massfrac, gdnv_state_p, gdnv_state_e);

  mask = (spout < 0.0);
  rgd = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    rspgd[n] = mask ? rspo[n] : rspgd[n];
    rgd += rspgd[n];
  }
  qint_iu = mask ? uo : qint_iu;
  qint_gdpres = mask ? po : qint_gdpres;

  mask = (spin >= 0.0);
  rgd = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    rspgd[n] = mask ? rspstar[n] : rspgd[n];
    rgd += rspgd[n];
  }
  qint_iu = mask ? ustar : qint_iu;
  qint_gdpres = mask ? pstar : qint_gdpres;

  gdnv_state_rho = rgd;
  gdnv_state_p = qint_gdpres;
  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = rspgd[n] / rgd;
  }
  RYP2E(gdnv_state_rho, gdnv_state_massfrac, gdnv_state_p, gdnv_state_e);
  double regd = gdnv_state_rho * gdnv_state_e;

  qint_gdgame = qint_gdpres / regd + 1.0;
  qint_iu = bc_test_val * qint_iu;
  uflx_rho = rgd * qint_iu;
  for (int n = 0; n < NUM_SPECIES; n++) {
    uflx_rhoY[n] = rspgd[n] * qint_iu;
  }
  uflx_u = uflx_rho * qint_iu + qint_gdpres;
  uflx_v = uflx_rho * qint_iv1;
  uflx_w = uflx_rho * qint_iv2;
  const double rhoetot =
    regd +
    0.5 * rgd * (qint_iu * qint_iu + qint_iv1 * qint_iv1 + qint_iv2 * qint_iv2);
  uflx_eden = qint_iu * (rhoetot + qint_gdpres);
  uflx_eint = qint_iu * regd;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pc_cmpflx(const int i, const int j, const int k, const int bclo, const int bchi, const int domlo, const int domhi,
	  const double * ql, const int ql_jstride, const int ql_kstride, const int ql_nstride, const int ql_beginx, const int ql_beginy, const int ql_beginz,
	  const double * qr, const int qr_jstride, const int qr_kstride, const int qr_nstride, const int qr_beginx, const int qr_beginy, const int qr_beginz,
	  double * flx, const int flx_jstride, const int flx_kstride, const int flx_nstride, const int flx_beginx, const int flx_beginy, const int flx_beginz,
	  double * q, const int q_jstride, const int q_kstride, const int q_nstride, const int q_beginx, const int q_beginy, const int q_beginz,
	  const double * qa, const int qa_jstride, const int qa_kstride, const int qa_nstride, const int qa_beginx, const int qa_beginy, const int qa_beginz,
	  const int dir)
{
  double cav, ustar;
  double spl[NUM_SPECIES];
  double spr[NUM_SPECIES];
  int idx;
  int IU, IV, IV2;
  int GU, GV, GV2;
  int f_idx[3];
  if (dir == 0) {
    IU = QU;
    IV = QV;
    IV2 = QW;
    GU = GDU;
    GV = GDV;
    GV2 = GDW;
    //cav = 0.5 * (qa(i, j, k, QC) + qa(i - 1, j, k, QC));
    cav = 0.5 * (qa[(i-qa_beginx)+(j-qa_beginy)*qa_jstride+(k-qa_beginz)*qa_kstride+QC*qa_nstride] +
		 qa[(i-1-qa_beginx)+(j-qa_beginy)*qa_jstride+(k-qa_beginz)*qa_kstride+QC*qa_nstride]);
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
    //cav = 0.5 * (qa(i, j, k, QC) + qa(i, j - 1, k, QC));
    cav = 0.5 * (qa[(i-qa_beginx)+(j-qa_beginy)*qa_jstride+(k-qa_beginz)*qa_kstride+QC*qa_nstride] +
		 qa[(i-qa_beginx)+(j-1-qa_beginy)*qa_jstride+(k-qa_beginz)*qa_kstride+QC*qa_nstride]);
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
    //cav = 0.5 * (qa(i, j, k, QC) + qa(i, j, k - 1, QC));
    cav = 0.5 * (qa[(i-qa_beginx)+(j-qa_beginy)*qa_jstride+(k-qa_beginz)*qa_kstride+QC*qa_nstride] +
		 qa[(i-qa_beginx)+(j-qa_beginy)*qa_jstride+(k-1-qa_beginz)*qa_kstride+QC*qa_nstride]);
    f_idx[0] = UMZ;
    f_idx[1] = UMX;
    f_idx[2] = UMY;
  }

  for (int sp = 0; sp < NUM_SPECIES; ++sp) {
    //spl[sp] = ql(i, j, k, QFS + sp);
    spl[sp] = ql[(i-ql_beginx)+(j-ql_beginy)*ql_jstride+(k-ql_beginz)*ql_kstride+(QFS + sp)*ql_nstride];
    //spr[sp] = qr(i, j, k, QFS + sp);
    spr[sp] = qr[(i-qr_beginx)+(j-qr_beginy)*qr_jstride+(k-qr_beginz)*qr_kstride+(QFS + sp)*qr_nstride];
  }

  //double ul = ql(i, j, k, IU);
  double ul = ql[(i-ql_beginx)+(j-ql_beginy)*ql_jstride+(k-ql_beginz)*ql_kstride+(IU)*ql_nstride];
  //double vl = ql(i, j, k, IV);
  double vl = ql[(i-ql_beginx)+(j-ql_beginy)*ql_jstride+(k-ql_beginz)*ql_kstride+(IV)*ql_nstride];
  //double v2l = ql(i, j, k, IV2);
  double v2l = ql[(i-ql_beginx)+(j-ql_beginy)*ql_jstride+(k-ql_beginz)*ql_kstride+(IV2)*ql_nstride];
  //double pl = ql(i, j, k, QPRES);
  double pl = ql[(i-ql_beginx)+(j-ql_beginy)*ql_jstride+(k-ql_beginz)*ql_kstride+(QPRES)*ql_nstride];
  //double rhol = ql(i, j, k, QRHO);
  double rhol = ql[(i-ql_beginx)+(j-ql_beginy)*ql_jstride+(k-ql_beginz)*ql_kstride+(QRHO)*ql_nstride];

  //double ur = qr(i, j, k, IU);
  double ur = qr[(i-qr_beginx)+(j-qr_beginy)*qr_jstride+(k-qr_beginz)*qr_kstride+(IU)*qr_nstride];
  //double vr = qr(i, j, k, IV);
  double vr = qr[(i-qr_beginx)+(j-qr_beginy)*qr_jstride+(k-qr_beginz)*qr_kstride+(IV)*qr_nstride];
  //double v2r = qr(i, j, k, IV2);
  double v2r = qr[(i-qr_beginx)+(j-qr_beginy)*qr_jstride+(k-qr_beginz)*qr_kstride+(IV2)*qr_nstride];
  //double pr = qr(i, j, k, QPRES);
  double pr = qr[(i-qr_beginx)+(j-qr_beginy)*qr_jstride+(k-qr_beginz)*qr_kstride+(QPRES)*qr_nstride];
  //double rhor = qr(i, j, k, QRHO);
  double rhor = qr[(i-qr_beginx)+(j-qr_beginy)*qr_jstride+(k-qr_beginz)*qr_kstride+(QRHO)*qr_nstride];

  // Boundary condition corrections
  if (dir == 2) {
    idx = k;
  } else {
    idx = (dir == 0) ? i : j;
  }
#if 0
  if (idx == domlo) {
    if (
      bclo == PCPhysBCType::no_slip_wall || bclo == PCPhysBCType::slip_wall ||
      bclo == PCPhysBCType::symmetry) {
      ul = -ur;
      vl = vr;
      v2l =
        v2r; // NoSlip: this is fine because Godunov velocity normal will be 0
      pl = pr;
      rhol = rhor;
    } else if (bclo == PCPhysBCType::outflow) {
      ul = ur;
      vl = vr;
      v2l = v2r;
      pl = pr;
      rhol = rhor;
    }
  } else if (idx == domhi + 1) {
    if (
      bchi == PCPhysBCType::no_slip_wall || bchi == PCPhysBCType::slip_wall ||
      bchi == PCPhysBCType::symmetry) {
      ur = -ul;
      vr = vl;
      v2r =
        v2l; // NoSlip: this is fine because Godunov velocity normal will be 0
      pr = pl;
      rhor = rhol;
    } else if (bchi == PCPhysBCType::outflow) {
      ur = ul;
      vr = vl;
      v2r = v2l;
      pr = pl;
      rhor = rhol;
    }
  }
#endif
  
  const int bc_test_val = 1;
  double dummy_flx[NUM_SPECIES] = {0.0};
  riemann(rhol, ul, vl, v2l, pl, spl, rhor, ur, vr, v2r, pr, spr, bc_test_val, cav, ustar,
	  flx[(i-flx_beginx)+(j-flx_beginy)*flx_jstride+(k-flx_beginz)*flx_kstride+(URHO)*flx_nstride], //flx(i, j, k, URHO),
	  dummy_flx,
	  flx[(i-flx_beginx)+(j-flx_beginy)*flx_jstride+(k-flx_beginz)*flx_kstride+(f_idx[0])*flx_nstride], //flx(i, j, k, f_idx[0]),
	  flx[(i-flx_beginx)+(j-flx_beginy)*flx_jstride+(k-flx_beginz)*flx_kstride+(f_idx[1])*flx_nstride], //flx(i, j, k, f_idx[1]),
	  flx[(i-flx_beginx)+(j-flx_beginy)*flx_jstride+(k-flx_beginz)*flx_kstride+(f_idx[2])*flx_nstride], //flx(i, j, k, f_idx[2]),
	  flx[(i-flx_beginx)+(j-flx_beginy)*flx_jstride+(k-flx_beginz)*flx_kstride+(UEDEN)*flx_nstride], //flx(i, j, k, UEDEN),
	  flx[(i-flx_beginx)+(j-flx_beginy)*flx_jstride+(k-flx_beginz)*flx_kstride+(UEINT)*flx_nstride], //flx(i, j, k, UEINT),
	  q[(i-q_beginx)+(j-q_beginy)*q_jstride+(k-q_beginz)*q_kstride+(GU)*q_nstride], //q(i, j, k, GU),
	  q[(i-q_beginx)+(j-q_beginy)*q_jstride+(k-q_beginz)*q_kstride+(GV)*q_nstride], //q(i, j, k, GV),
	  q[(i-q_beginx)+(j-q_beginy)*q_jstride+(k-q_beginz)*q_kstride+(GV2)*q_nstride], //q(i, j, k, GV2),
	  q[(i-q_beginx)+(j-q_beginy)*q_jstride+(k-q_beginz)*q_kstride+(GDPRES)*q_nstride], //q(i, j, k, GDPRES),
	  q[(i-q_beginx)+(j-q_beginy)*q_jstride+(k-q_beginz)*q_kstride+(GDGAME)*q_nstride] //q(i, j, k, GDGAME)
	  );
  
  //double flxrho = flx(i, j, k, URHO);
  double flxrho = flx[(i-flx_beginx)+(j-flx_beginy)*flx_jstride+(k-flx_beginz)*flx_kstride+(URHO)*flx_nstride];
  //const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
#if NUM_ADV > 0
  for (int n = 0; n < NUM_ADV; n++) {
    const int qc = QFA + n;
    pc_cmpflx_passive(ustar, flxrho, ql(iv, qc), qr(iv, qc), flx(iv, UFA + n));
  }
#endif
  for (int n = 0; n < NUM_SPECIES; n++) {
    const int qc = QFS + n;    
    pc_cmpflx_passive(ustar, flxrho,
		      ql[(i-ql_beginx)+(j-ql_beginy)*ql_jstride+(k-ql_beginz)*ql_kstride+(qc)*ql_nstride], //ql(iv, qc)
		      qr[(i-qr_beginx)+(j-qr_beginy)*qr_jstride+(k-qr_beginz)*qr_kstride+(qc)*qr_nstride], //qr(iv, qc)
		      flx[(i-flx_beginx)+(j-flx_beginy)*flx_jstride+(k-flx_beginz)*flx_kstride+(UFS + n)*flx_nstride]); //flx(iv, UFS + n));
  }
#if NUM_AUX > 0
  for (int n = 0; n < NUM_AUX; n++) {
    const int qc = QFX + n;
    pc_cmpflx_passive(ustar, flxrho, ql(iv, qc), qr(iv, qc), flx(iv, UFX + n));
  }
#endif
#if NUM_LIN > 0
  for (int n = 0; n < NUM_LIN; n++) {
    const int qc = QLIN + n;
    pc_cmpflx_passive(
      ustar, q(i, j, k, GU), ql(iv, qc), qr(iv, qc), flx(iv, ULIN + n));
  }
#endif
}

__global__ void
pc_cmpflx_launch(const int bclo, const int bchi, const int domlo, const int domhi, const int ncells, const int lenx, const int lenxy, const int lox, const int loy, const int loz,
		 const double * qlxy, const int qlxy_jstride, const int qlxy_kstride, const int qlxy_nstride, const int qlxy_beginx, const int qlxy_beginy, const int qlxy_beginz,
		 const double * qrxy, const int qrxy_jstride, const int qrxy_kstride, const int qrxy_nstride, const int qrxy_beginx, const int qrxy_beginy, const int qrxy_beginz,
		 double * flxy, const int flxy_jstride, const int flxy_kstride, const int flxy_nstride, const int flxy_beginx, const int flxy_beginy, const int flxy_beginz,
		 double * qxy, const int qxy_jstride, const int qxy_kstride, const int qxy_nstride, const int qxy_beginx, const int qxy_beginy, const int qxy_beginz,
		 const double * qlxz, const int qlxz_jstride, const int qlxz_kstride, const int qlxz_nstride, const int qlxz_beginx, const int qlxz_beginy, const int qlxz_beginz,
		 const double * qrxz, const int qrxz_jstride, const int qrxz_kstride, const int qrxz_nstride, const int qrxz_beginx, const int qrxz_beginy, const int qrxz_beginz,
		 double * flxz, const int flxz_jstride, const int flxz_kstride, const int flxz_nstride, const int flxz_beginx, const int flxz_beginy, const int flxz_beginz,
		 double * qxz, const int qxz_jstride, const int qxz_kstride, const int qxz_nstride, const int qxz_beginx, const int qxz_beginy, const int qxz_beginz,
		 const double * qaux, const int qaux_jstride, const int qaux_kstride, const int qaux_nstride, const int qaux_beginx, const int qaux_beginy, const int qaux_beginz,
		 const int dir)
{

  for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
       icell < ncells; icell += stride)
    {
      int k =  icell /   lenxy;
      int j = (icell - k*lenxy) /   lenx;
      int i = (icell - k*lenxy) - j*lenx;
      i += lox;
      j += loy;
      k += loz;
       
      // X|Y
      pc_cmpflx(i, j, k, bclo, bchi, domlo, domhi,
		qlxy, qlxy_jstride, qlxy_kstride, qlxy_nstride, qlxy_beginx, qlxy_beginy, qlxy_beginz,
		qrxy, qrxy_jstride, qrxy_kstride, qrxy_nstride, qrxy_beginx, qrxy_beginy, qrxy_beginz,
		flxy, flxy_jstride, flxy_kstride, flxy_nstride, flxy_beginx, flxy_beginy, flxy_beginz,
		qxy, qxy_jstride, qxy_kstride, qxy_nstride, qxy_beginx, qxy_beginy, qxy_beginz,
		qaux, qaux_jstride, qaux_kstride, qaux_nstride, qaux_beginx, qaux_beginy, qaux_beginz,
		dir);
      //pc_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qmxy, qpxy, flxy, qxy, qaux, cdir);
      // X|Z
      pc_cmpflx(i, j, k, bclo, bchi, domlo, domhi,
		qlxz, qlxz_jstride, qlxz_kstride, qlxz_nstride, qlxz_beginx, qlxz_beginy, qlxz_beginz,
		qrxz, qrxz_jstride, qrxz_kstride, qrxz_nstride, qrxz_beginx, qrxz_beginy, qrxz_beginz,
		flxz, flxz_jstride, flxz_kstride, flxz_nstride, flxz_beginx, flxz_beginy, flxz_beginz,
		qxz, qxz_jstride, qxz_kstride, qxz_nstride, qxz_beginx, qxz_beginy, qxz_beginz,
		qaux, qaux_jstride, qaux_kstride, qaux_nstride, qaux_beginx, qaux_beginy, qaux_beginz,
		dir);
      //pc_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qmxz, qpxz, flxz, qxz, qaux, cdir);
    }
}

#include <thrust/count.h>
#include <thrust/execution_policy.h>

struct is_bad
{
  __host__ __device__
  bool operator()(double &x)
  {
    return isinf(x) || isnan(x);
  }
};

void checkBadValues(double * p, size_t N, std::string Name, int _LINE_)
{
  auto result = thrust::count_if(thrust::device, p, p+N, is_bad());
  if (result) printf("found %ld bad values in %s at line %d\n",result,Name.c_str(),_LINE_);
}

void readFillData(std::string path_to, std::string name, double * dbuffer, size_t& size,
		  int& nComp, int &jstride, int& kstride, int& nstride,
		  int& beginx, int& beginy, int& beginz)
{
  int rank = 0;
  std::fstream csvfile;
  char fname[100];
  sprintf(fname,"%s/%s/%s_metadata_%s_%d.csv",path_to.c_str(),NAME,NAME,name.c_str(),rank);
  csvfile.open(fname, std::ios::in);
  std::string tp;
  getline(csvfile, tp);
  getline(csvfile, tp);
  std::string s=tp;
  std::string delimiter = ", ";
  size_t pos = 0;
  std::string token;
  std::vector<std::string> strs(0);
  while ((pos = s.find(delimiter)) != std::string::npos) {
    token = s.substr(0, pos);
    strs.push_back(token);
    s.erase(0, pos + delimiter.length());
  }
  strs.push_back(s);

  /* read data */
  size = (size_t) std::stoi(strs[0]);
  nComp = std::stoi(strs[1]);
  jstride = std::stoi(strs[2]);
  kstride = std::stoi(strs[3]);
  nstride = std::stoi(strs[4]);
  beginx  = std::stoi(strs[5]);
  beginy  = std::stoi(strs[6]);
  beginz  = std::stoi(strs[7]);
#ifdef DEBUG
  std::cout << "  qmxy:\n\tsize=" << size << " nComp=" << nComp << std::endl;
  std::cout << "\tjstride=" << jstride << " kstride=" << kstride << " nstride=" << nstride << std::endl;
  std::cout << "\tbeginx=" << beginx << " beginy=" << beginy << " beginz=" << beginz << std::endl;
#endif
  std::vector<double> hbuffer(size);

  sprintf(fname,"%s/%s/%s_%s_rank_%d.bin",path_to.c_str(),NAME,NAME,name.c_str(),rank);
  FILE * fid = fopen(fname,"rb");
  fread(hbuffer.data(), sizeof(double), size, fid);
  fclose(fid);
  HIP_CALL(hipMemcpy(dbuffer, hbuffer.data(), sizeof(double) * size, hipMemcpyHostToDevice));
#ifdef DEBUG
  printf("\tInitialized %s!\n",name.c_str()); fflush(stdout);
#endif
}


void writeToFile(std::string path_to, std::string name, double * dbuffer, size_t size)
{
  int rank=0;
  std::vector<double> temp(size);
  HIP_CALL(hipMemcpy(temp.data(), dbuffer, sizeof(double) * size, hipMemcpyDeviceToHost));
  
  char fname[100];
  sprintf(fname,"%s/%s/%s_%s_rank_%d.bin",path_to.c_str(),NAME,NAME,name.c_str(),rank);
  FILE * fid = fopen(fname,"wb");
  fwrite(temp.data(), sizeof(double), size, fid);
  fclose(fid);
}

void checkResults(std::string path_to, std::string name, double * dbuffer, size_t size, double rtol, double atol)
{
  int rank=0;
  std::vector<double> temp(size);
  HIP_CALL(hipMemcpy(temp.data(), dbuffer, sizeof(double) * size, hipMemcpyDeviceToHost));
  
  std::vector<double> pele(size);
  char fname[100];
  sprintf(fname,"%s/%s/%s_%s_rank_%d.bin",path_to.c_str(),NAME,NAME,name.c_str(),rank);
  FILE * fid = fopen(fname,"rb");
  fread(pele.data(), sizeof(double), size, fid);
  fclose(fid);

  /* write some checking code */
  bool failure = false;
  int count=0;
  for (int i=0; i<size; ++i)
    {
      double a = temp[i];
      double b = pele[i];
      bool isclose = std::abs(a-b) <= std::max(rtol*max(std::abs(a),std::abs(b)), atol);
      if (!isclose || !std::isfinite(a)  || !std::isfinite(b))
	{
	  failure = true;
	  if (count<10)
	    printf("\ti=%d : Pele=%1.15g, repro=%1.15g\n",i,pele[i],temp[i]);
	  count++;
	}
      else if (count<20)
	{
	  //printf("\ti=%d : Pele=%1.15g, repro=%1.15g\n",i,pele[i],temp[i]);
	  //count++;
	}
    }
  if (failure) std::cout << name << " has " << count << " values that are NOT close, |repro-Pele| <= std::max(rtol*max(|repro|,|Pele|), atol), with rtol="
			 << rtol << " atol=" << atol <<  std::endl;

}
    
  

/****************************************************************/
/* main                                                         */
/****************************************************************/

int main(int argc, char * argv[])
{
  std::string input_path_to = "./";
  if (argc>=2)
    input_path_to = std::string(argv[1]);

  std::string output_path_to = "./";
  if (argc>=3)
    output_path_to = std::string(argv[2]);

  std::string comp_path_to = "./";
  if (argc>=4)
    comp_path_to = std::string(argv[3]);

  std::cout << "input_path_to=" << input_path_to << std::endl;
  std::cout << "output_path_to=" << output_path_to << std::endl;
  std::cout << "comp_path_to=" << comp_path_to << std::endl;

  int poolSize = 10;
  if (argc>=5)
    poolSize = atoi(argv[4]);

  float rtol = 1.e-5;
  float atol = 1.e-8;
  if (argc>=6)
    rtol = atof(argv[5]);
  if (argc>=7)
    atol = atof(argv[6]);
  
  int rank = 0;

  std::fstream csvfile;
  char fname[100];
  sprintf(fname,"%s/%s/%s_metadata_repro2_%d.csv",input_path_to.c_str(),NAME,NAME,rank);
  csvfile.open(fname, std::ios::in);
  std::string tp;
  getline(csvfile, tp);
  getline(csvfile, tp);
  std::string s=tp;
  std::string delimiter = ", ";
  size_t pos = 0;
  std::string token;
  std::vector<std::string> strs(0);
  while ((pos = s.find(delimiter)) != std::string::npos) {
    token = s.substr(0, pos);
    strs.push_back(token);
    s.erase(0, pos + delimiter.length());
  }
  strs.push_back(s);
  int bclo = std::stoi(strs[0]);
  int bchi = std::stoi(strs[1]);
  int dlx = std::stoi(strs[2]);
  int dhx = std::stoi(strs[3]);
  int cdir = std::stoi(strs[4]);

  int ncells = std::stoi(strs[5]);
  int lenx   = std::stoi(strs[6]);
  int lenxy  = std::stoi(strs[7]);
  int lox    = std::stoi(strs[8]);
  int loy    = std::stoi(strs[9]);
  int loz    = std::stoi(strs[10]);
  csvfile.close();

#ifdef DEBUG
  std::cout << "\tbclo=" << bclo << std::endl;
  std::cout << "\tbchi=" << bchi << std::endl;
  std::cout << "\tdlx=" << dlx << std::endl;
  std::cout << "\tdhx=" << dhx << std::endl;
  std::cout << "\tcdir=" << cdir << std::endl;
  std::cout << "\tncells=" << ncells << std::endl;
  std::cout << "\tlenx=" << lenx << " lenxy=" << lenxy << std::endl;
  std::cout << "\tlox=" << lox << " loy=" << loy << " loz=" << loz << std::endl;
#endif

  hipStream_t stream=0;
  //HIP_CALL(hipStreamCreate(&stream));
  double * pool;
  size_t e = poolSize*1e9;
  HIP_CALL(hipMalloc((void **)&pool, e));

  double * qmxy, * qpxy, * flxy, * qxy, * qmxz, * qpxz, * flxz, * qxz, * qaux;

  /* qmxy */
  qmxy = pool;
  size_t size_qmxy;
  int nComp_qmxy, jstride_qmxy, kstride_qmxy, nstride_qmxy,  beginx_qmxy, beginy_qmxy, beginz_qmxy;
  readFillData(input_path_to, "qmxy", qmxy, size_qmxy, nComp_qmxy, 
	       jstride_qmxy, kstride_qmxy, nstride_qmxy,
	       beginx_qmxy, beginy_qmxy, beginz_qmxy);
  
  /* qpxy */
  qpxy = qmxy + size_qmxy*sizeof(double);
  size_t size_qpxy;
  int nComp_qpxy, jstride_qpxy, kstride_qpxy, nstride_qpxy,  beginx_qpxy, beginy_qpxy, beginz_qpxy;
  readFillData(input_path_to, "qpxy", qpxy, size_qpxy, nComp_qpxy, 
	       jstride_qpxy, kstride_qpxy, nstride_qpxy,
	       beginx_qpxy, beginy_qpxy, beginz_qpxy);

  /* flxy */
  flxy = qpxy + size_qpxy*sizeof(double);
  size_t size_flxy;
  int nComp_flxy, jstride_flxy, kstride_flxy, nstride_flxy,  beginx_flxy, beginy_flxy, beginz_flxy;
  readFillData(input_path_to, "flxy", flxy, size_flxy, nComp_flxy, 
	       jstride_flxy, kstride_flxy, nstride_flxy,
	       beginx_flxy, beginy_flxy, beginz_flxy);

  /* qxy */
  qxy  = flxy + size_flxy*sizeof(double);
  size_t size_qxy;
  int nComp_qxy, jstride_qxy, kstride_qxy, nstride_qxy,  beginx_qxy, beginy_qxy, beginz_qxy;
  readFillData(input_path_to, "qxy", qxy, size_qxy, nComp_qxy, 
	       jstride_qxy, kstride_qxy, nstride_qxy,
	       beginx_qxy, beginy_qxy, beginz_qxy);

  /* qmxz */
  qmxz = qxy + size_qxy*sizeof(double);
  size_t size_qmxz;
  int nComp_qmxz, jstride_qmxz, kstride_qmxz, nstride_qmxz,  beginx_qmxz, beginy_qmxz, beginz_qmxz;
  readFillData(input_path_to, "qmxz", qmxz, size_qmxz, nComp_qmxz, 
	       jstride_qmxz, kstride_qmxz, nstride_qmxz,
	       beginx_qmxz, beginy_qmxz, beginz_qmxz);

  /* qpxz */
  qpxz = qmxz + size_qmxz*sizeof(double);
  size_t size_qpxz;
  int nComp_qpxz, jstride_qpxz, kstride_qpxz, nstride_qpxz,  beginx_qpxz, beginy_qpxz, beginz_qpxz;
  readFillData(input_path_to, "qpxz", qpxz, size_qpxz, nComp_qpxz, 
	       jstride_qpxz, kstride_qpxz, nstride_qpxz,
	       beginx_qpxz, beginy_qpxz, beginz_qpxz);

  /* flxz */
  flxz = qpxz + size_qpxz*sizeof(double);
  size_t size_flxz;
  int nComp_flxz, jstride_flxz, kstride_flxz, nstride_flxz,  beginx_flxz, beginy_flxz, beginz_flxz;
  readFillData(input_path_to, "flxz", flxz, size_flxz, nComp_flxz, 
	       jstride_flxz, kstride_flxz, nstride_flxz,
	       beginx_flxz, beginy_flxz, beginz_flxz);

  /* qxz */
  qxz  = flxz + size_flxz*sizeof(double);
  size_t size_qxz;
  int nComp_qxz, jstride_qxz, kstride_qxz, nstride_qxz,  beginx_qxz, beginy_qxz, beginz_qxz;
  readFillData(input_path_to, "qxz", qxz, size_qxz, nComp_qxz, 
	       jstride_qxz, kstride_qxz, nstride_qxz,
	       beginx_qxz, beginy_qxz, beginz_qxz);

  /* qmxz */
  qaux = qxz + size_qxz*sizeof(double);
  size_t size_qaux;
  int nComp_qaux, jstride_qaux, kstride_qaux, nstride_qaux,  beginx_qaux, beginy_qaux, beginz_qaux;
  readFillData(input_path_to, "qaux", qaux, size_qaux, nComp_qaux, 
	       jstride_qaux, kstride_qaux, nstride_qaux,
	       beginx_qaux, beginy_qaux, beginz_qaux);

#if 0
  HIP_CALL(hipMemset(flxy, 0, size_flxy*sizeof(double)));
  HIP_CALL(hipMemset(flxz, 0, size_flxz*sizeof(double)));
  HIP_CALL(hipMemset(qxy, 0, size_qxy*sizeof(double)));
  HIP_CALL(hipMemset(qxz, 0, size_qxz*sizeof(double)));
#endif  
  checkBadValues(qmxy, size_qmxy, "qmxy", __LINE__);
  checkBadValues(qpxy, size_qpxy, "qpxy", __LINE__);
  checkBadValues(flxy, size_flxy, "flxy", __LINE__);
  checkBadValues(qxy, size_qxy, "qxy", __LINE__);

  checkBadValues(qmxz, size_qmxz, "qmxz", __LINE__);
  checkBadValues(qpxz, size_qpxz, "qpxz", __LINE__);
  checkBadValues(flxz, size_flxz, "flxz", __LINE__);
  checkBadValues(qxz, size_qxz, "qxz", __LINE__);
  checkBadValues(qaux, size_qaux, "qaux", __LINE__);

  HIP_CALL(hipGetLastError());
  HIP_CALL(hipStreamSynchronize(stream));

  const int nthreads=256;
  const int nblocks = (ncells+nthreads-1)/nthreads;
  pc_cmpflx_launch<<<nblocks, nthreads>>>(bclo, bchi, dlx, dhx, ncells, lenx, lenxy, lox, loy, loz,
					  qmxy, jstride_qmxy, kstride_qmxy, nstride_qmxy, beginx_qmxy, beginy_qmxy, beginz_qmxy,
					  qpxy, jstride_qpxy, kstride_qpxy, nstride_qpxy, beginx_qpxy, beginy_qpxy, beginz_qpxy,
					  flxy, jstride_flxy, kstride_flxy, nstride_flxy, beginx_flxy, beginy_flxy, beginz_flxy,
					  qxy, jstride_qxy, kstride_qxy, nstride_qxy, beginx_qxy, beginy_qxy, beginz_qxy,
					  qmxz, jstride_qmxz, kstride_qmxz, nstride_qmxz, beginx_qmxz, beginy_qmxz, beginz_qmxz,
					  qpxz, jstride_qpxz, kstride_qpxz, nstride_qpxz, beginx_qpxz, beginy_qpxz, beginz_qpxz,
					  flxz, jstride_flxz, kstride_flxz, nstride_flxz, beginx_flxz, beginy_flxz, beginz_flxz,
					  qxz, jstride_qxz, kstride_qxz, nstride_qxz, beginx_qxz, beginy_qxz, beginz_qxz,
					  qaux, jstride_qaux, kstride_qaux, nstride_qaux, beginx_qaux, beginy_qaux, beginz_qaux,
					  cdir);
  
  HIP_CALL(hipGetLastError());
  HIP_CALL(hipStreamSynchronize(stream));

  checkBadValues(flxy, size_flxy, "flxy", __LINE__);
  checkBadValues(qxy, size_qxy, "qxy", __LINE__);
  checkBadValues(flxz, size_flxz, "flxz", __LINE__);
  checkBadValues(qxz, size_qxz, "qxz", __LINE__);

  writeToFile(output_path_to, "flxy", flxy, size_flxy);
  writeToFile(output_path_to, "flxz", flxz, size_flxz);
  writeToFile(output_path_to, "qxy", qxy, size_qxy);
  writeToFile(output_path_to, "qxz", qxz, size_qxz);
  
  checkResults(comp_path_to, "flxy", flxy, size_flxy, rtol, atol);
  checkResults(comp_path_to, "flxz", flxz, size_flxz, rtol, atol);
  checkResults(comp_path_to, "qxy", qxy, size_qxy, rtol, atol);
  checkResults(comp_path_to, "qxz", qxz, size_qxz, rtol, atol);

  /* Cleanup */
  HIP_CALL(hipFree(pool));
  //HIP_CALL(hipStreamDestroy(stream));
  return 0;
}

