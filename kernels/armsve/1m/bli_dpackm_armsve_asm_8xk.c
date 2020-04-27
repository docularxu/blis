/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include "stdio.h"  // TODO: to be deleted

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#else
#error "No Arm SVE intrinsics support in compiler"
#endif /* __ARM_FEATURE_SVE */

#if 1
void print_f64_array ( double * p )
{
    int i;

    printf("in %s(): \n", __func__);
    for ( i = 0; i < 4; i ++ )
	    printf("%f,\t", *(p+i));
    printf("\n");
    return;
}

void print_uint64_vector( svuint64_t z_index )
{
    const svbool_t  all_active = svptrue_b64();
    uint64_t  elem[4];
    int i;

    printf("in %s(): \n", __func__);
    svst1_u64( all_active, elem, z_index);
    for ( i = 0; i < 4; i ++ )
	    printf("%u,\t", elem[i]);
    printf("\n");
    return;
}

void print_vector ( svfloat64_t z_f64 )
{
    const svbool_t  all_active = svptrue_b64();
    float64_t  elem[4];
    int i;

    printf("in %s(): \n", __func__);
    svst1_f64( all_active, elem, z_f64);
    for ( i = 0; i < 4; i ++ )
	    printf("%f,\t", elem[i]);
    printf("\n");
    return; 
}
#endif

/* assumption:
 *   SVE vector length = 256 bits.
 */

void bli_dpackm_armsve256_asm_8xk
     (
       conj_t           conja,
       pack_t           schema,
       dim_t            cdim_,
       dim_t            n_,
       dim_t            n_max_,
       void*   restrict kappa_,
       void*   restrict a_, inc_t inca_, inc_t lda_,
       void*   restrict p_,              inc_t ldp_,
       cntx_t* restrict cntx
     )
{
    double*       a     = ( double* )a_;
    double*       p     = ( double* )p_;
    double*       kappa = ( double* )kappa_;
    const int64_t cdim  = cdim_;
    const int64_t mnr   = 8;
    const int64_t n     = n_;
    const int64_t n_max = n_max_;
    const int64_t inca  = inca_;
    const int64_t lda   = lda_;
    const int64_t ldp   = ldp_;

#if 1
  	printf("PACK: in armsve256: %s() \n", __func__);
	  printf("          conja=%d; schema=%d\n", conja, schema);
	  printf("          cdim=%d; n=%d; n_max=%d\n", cdim, n, n_max);
    printf("          kappa=%f\n", *kappa);
	  printf("          a=0x%x; inca=%d, lda=%d\n", a, inca, lda);
	  printf("          p=0x%x, ldp=%d; cntx=0x%x\n", p, ldp, cntx);
#endif

    const svbool_t   all_active = svptrue_b64();
    svfloat64_t      z_a0; // a( 0:3,x );
    svfloat64_t      z_a4; // a( 4:7,x );

    // creating index for gather/scatter
    //   with each element as: 0, 1*inca, 2*inca, 3*inca
    svuint64_t  z_index;
    z_index = svindex_u64( 0, inca * sizeof(double) );

#if 1
    print_uint64_vector( z_index );
#endif

    double* restrict alpha1     = a;
    double* restrict pi1        = p;
    double* restrict alpha1_4   = alpha1 + 4 * inca;

  if ( cdim == mnr )
  {
      if ( bli_deq1( *kappa ) )
      {
        if ( inca == 1 )  // continous memory. packA style
        {
          printf(" in condition (cdim == mnr) && (*kappa == 1.0) && (inca ==  1)\n");
	  for ( dim_t k = n; k != 0; --k )
	  {
            // load 8 continuous elments from *a
            // z_a0 = svld1_f64( all_active, alpha1 );
            // z_a4 = svld1_vnum_f64( all_active, alpha1, 1 );
#if 1
            // gather load from *a
            z_a0 = svld1_gather_u64offset_f64( all_active, alpha1, z_index );
            z_a4 = svld1_gather_u64offset_f64( all_active, alpha1_4, z_index );
	    print_f64_array(alpha1);
	    print_vector(z_a0);
	    print_f64_array(alpha1+4);
	    print_vector(z_a4);
#endif
	    // store them into *p
            svst1_f64( all_active, pi1, z_a0 );
            svst1_vnum_f64( all_active, pi1, 1, z_a4 );

	    alpha1 += lda;
            alpha1_4 = alpha1 + 4 * inca;
	    pi1    += ldp;
	  }
#if 0
// TODO: unloop @ 4
	dim_t           n_iter     = n / 4;
	dim_t           n_left     = n % 4;
  			  for ( ??? ; n_iter != 0; --n_iter )
					{ 
            ... ...
            alpha1 += 2*lda;
  					pi1    += 2*ldp;
				  }
				  for ( ; n_left != 0; --n_left )
				  {
			  		PASTEMAC(ch,copys)( *(alpha1 + 0*inca), *(pi1 + 0) );
  					alpha1 += lda;
		  			pi1    += ldp;
				  }
#endif
        }
        else  // gather/scatter load/store. packB style
        {
          printf(" in condition (cdim == mnr) && (*kappa == 1.0) && (inca !=  1)\n");
				  for ( dim_t k = n; k != 0; --k )
				  {
            // gather load from *a
            z_a0 = svld1_gather_u64offset_f64( all_active, alpha1, z_index );
            z_a4 = svld1_gather_u64offset_f64( all_active, alpha1_4, z_index );
#if 1
	    print_vector(z_a0);
	    print_vector(z_a4);
#endif
            // scatter store into *p
            svst1_f64( all_active, pi1, z_a0 );
            svst1_vnum_f64( all_active, pi1, 1, z_a4 );

					  alpha1 += lda;
            alpha1_4 = alpha1 + 4 * inca;
					  pi1    += ldp;
				  }
        }
      }
      else  /* *kappa != 1.0 */
      /* multiply Kappa, between load and store */
      {
        /* load kappa into vector */
        svfloat64_t     z_kappa;
        z_kappa = svdup_f64( *kappa );

        if ( inca == 1 )  // continous memory. packA style
		    {
          printf(" in condition (cdim == mnr) && (*kappa != 1.0) && (inca ==  1)\n");
				  for ( dim_t k = n; k != 0; --k )
				  {
            // load 8 continuous elments from *a
					  z_a0 = svld1_f64( all_active, alpha1 );
            z_a4 = svld1_vnum_f64( all_active, alpha1, 1 );
            // multiply by *kappa
            z_a0 = svmul_lane_f64( z_a0, z_kappa, 0 );
            z_a4 = svmul_lane_f64( z_a4, z_kappa, 0 );
            // store them into *p
            svst1_f64( all_active, pi1, z_a0 );
            svst1_vnum_f64( all_active, pi1, 1, z_a4 );

					  alpha1 += lda;
					  pi1    += ldp;
				  }
        }
        else  // gather/scatter load/store. packB style
        {
          printf(" in condition (cdim == mnr) && (*kappa != 1.0) && (inca !=  1)\n");
				  for ( dim_t k = n; k != 0; --k )
				  {
            // gather load from *a
            z_a0 = svld1_gather_u64offset_f64( all_active, alpha1, z_index );
            z_a4 = svld1_gather_u64offset_f64( all_active, alpha1_4, z_index );
            // multiply by *kappa
            z_a0 = svmul_lane_f64( z_a0, z_kappa, 0 );
            z_a4 = svmul_lane_f64( z_a4, z_kappa, 0 );
            // scatter store into *p
            svst1_f64( all_active, pi1, z_a0 );
            svst1_vnum_f64( all_active, pi1, 1, z_a4 );

					  alpha1 += lda;
            alpha1_4 = alpha1 + 4 * inca;
					  pi1    += ldp;
				  }
        }
      } // end of if ( *kappa == 1.0 )
  }
	else // if ( cdim < mnr )
	{
          printf(" in condition (cdim < mnr) \n");
		bli_dscal2m_ex \
		( \
		  0, \
		  BLIS_NONUNIT_DIAG, \
		  BLIS_DENSE, \
		  ( trans_t )conja, \
		  cdim, \
		  n, \
		  kappa, \
		  a, inca, lda, \
		  p, 1,    ldp, \
		  cntx, \
		  NULL  \
		); \

		// if ( cdim < mnr )
		{
			const dim_t      i      = cdim;
			const dim_t      m_edge = mnr - i;
			const dim_t      n_edge = n_max;
			double* restrict p_edge = p + (i  )*1;

			bli_dset0s_mxn
			(
			  m_edge,
			  n_edge,
			  p_edge, 1, ldp
			);
		}
	}

	if ( n < n_max )
	{
		const dim_t      j      = n;
		const dim_t      m_edge = mnr;
		const dim_t      n_edge = n_max - j;
		double* restrict p_edge = p + (j  )*ldp;

		bli_dset0s_mxn
		(
		  m_edge,
		  n_edge,
		  p_edge, 1, ldp
		);
	}
}
