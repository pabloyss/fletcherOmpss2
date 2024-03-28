#include "ompss2_propagate.h"
#include "../derivatives.h"
#include "../map.h"


// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time
#pragma oss task in (pc[0:(sx*sy*sz)-1],qc[0:(sx*sy*sz)-1]) inout(pp[0:(sx*sy*sz)-1],qp[0:(sx*sy*sz)-1])
void OPENMP_Propagate(int sx, int sy, int sz, int bord,
                      float dx, float dy, float dz, float dt, int it,
                      float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc) {


#define SAMPLE_PRE_LOOP
#include "../sample.h"
#undef SAMPLE_PRE_LOOP


        // solve both equations in all internal grid points,
        // including absortion zone
    //#pragma oss task
// #pragma oss task in({pc[i],i=bord;sz-bord},{qc[i],i=bord;sz-bord},{ch1dxx[i],i=bord;sz-bord},{ch1dyy[i],i=bord;sz-bord},{ch1dzz[i],i=bord;sz-bord},{ch1dxy[i],i=bord;sz-bord},{ch1dyz[i],i=bord;sz-bord},{ch1dxz[i],i=bord;sz-bord},{v2sz[i],i=bord;sz-bord},{v2pz[i],i=bord;sz-bord},{v2px[i],i=bord;sz-bord},{v2pn[i],i=bord;sz-bord}) inout ({pp[i],i=bord;sz-bord}, {qp[i], i=bord;sz-bord})
    {
// #pragma oss taskloop grainsize (2) in (pc[iz],qc[iz],ch1dxx[iz],ch1dyy[iz],ch1dzz[iz],ch1dxy[iz],ch1dyz[iz],ch1dxz[iz],v2sz[iz],v2pz[iz],v2px[iz],v2pn[iz]) inout (pp[iz], qp[iz])
    #pragma oss taskloop
    for (int iz=bord; iz<sz-bord; iz++) {
            for (int iy=bord; iy<sy-bord; iy++) {
                for (int ix=bord; ix<sx-bord; ix++) {

#define SAMPLE_LOOP

                    const int i=ind(ix,iy,iz);

                    const float pxx= Der2(pc, i, strideX, dxxinv);
                    const float pyy= Der2(pc, i, strideY, dyyinv);
                    const float pzz= Der2(pc, i, strideZ, dzzinv);
                    const float pxy= DerCross(pc, i, strideX, strideY, dxyinv);
                    const float pyz= DerCross(pc, i, strideY, strideZ, dyzinv);
                    const float pxz= DerCross(pc, i, strideX, strideZ, dxzinv);

                    const float cpxx=ch1dxx[i]*pxx;
                    const float cpyy=ch1dyy[i]*pyy;
                    const float cpzz=ch1dzz[i]*pzz;
                    const float cpxy=ch1dxy[i]*pxy;
                    const float cpxz=ch1dxz[i]*pxz;
                    const float cpyz=ch1dyz[i]*pyz;
                    const float h1p=cpxx+cpyy+cpzz+cpxy+cpxz+cpyz;
                    const float h2p=pxx+pyy+pzz-h1p;

                    const float qxx= Der2(qc, i, strideX, dxxinv);
                    const float qyy= Der2(qc, i, strideY, dyyinv);
                    const float qzz= Der2(qc, i, strideZ, dzzinv);
                    const float qxy= DerCross(qc, i, strideX,  strideY, dxyinv);
                    const float qyz= DerCross(qc, i, strideY,  strideZ, dyzinv);
                    const float qxz= DerCross(qc, i, strideX,  strideZ, dxzinv);

                    const float cqxx=ch1dxx[i]*qxx;
                    const float cqyy=ch1dyy[i]*qyy;
                    const float cqzz=ch1dzz[i]*qzz;
                    const float cqxy=ch1dxy[i]*qxy;
                    const float cqxz=ch1dxz[i]*qxz;
                    const float cqyz=ch1dyz[i]*qyz;

                    const float h1q=cqxx+cqyy+cqzz+cqxy+cqxz+cqyz;
                    const float h2q=qxx+qyy+qzz-h1q;

                    const float h1pmq=h1p-h1q;
                    const float h2pmq=h2p-h2q;

                    const float rhsp=v2px[i]*h2p + v2pz[i]*h1q + v2sz[i]*h1pmq;
                    const float rhsq=v2pn[i]*h2p + v2pz[i]*h1q - v2sz[i]*h2pmq;

                    pp[i]=2.0f*pc[i] - pp[i] + rhsp*dt*dt;
                    qp[i]=2.0f*qc[i] - qp[i] + rhsq*dt*dt;


// END ONE SAMPLE
#undef SAMPLE_LOOP

                }
                }
            }
        }
#pragma oss taskwait

}
