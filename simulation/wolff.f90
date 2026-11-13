!======================================================================
!!======================================================================
! 2D Random-bond Ising model (quenched disorder)
! Wolff cluster Monte Carlo
!
! Parallelization model:
!   - ONE disorder realisation per job
!   - Realisation index read from command line
!
! Usage:
!   ./wolff_ising <realisation_id>
!
! OUTPUTS per (rid, T):
!   - obs.dat          (ASCII)
!   - correlation.dat  (ASCII)
!   - meta.txt         (ASCII metadata)
!   - output.dat       (ASCII per-configuration neighborhood counts,
!                       RAW format: one line per configuration with
!                       nplus(0) nminus(0) ... nplus(15) nminus(15))
!
! This version DOES NOT CREATE Jx.dat / Jy.dat.
! It EXPECTS them to already exist in the realisation folder and reads them.
!======================================================================

program wolff_random_bond_ising
  implicit none
  integer, parameter :: dp = selected_real_kind(15, 300)

  !----------------------- Lattice -----------------------
  integer, parameter :: L  = 50
  integer, parameter :: N  = L*L
  integer, parameter :: RMAX = L/2

  !----------------------- Disorder ----------------------
  real(dp), parameter :: J0    = 1.0_dp
  real(dp), parameter :: delta = 0.5_dp

  !----------------------- Temperatures ------------------
  real(dp), parameter :: T_min = 1.600_dp
  real(dp), parameter :: T_max = 2.400_dp
  real(dp), parameter :: dT    = 0.04_dp

  !----------------------- Wolff MC ----------------------
  integer, parameter :: THERM_CLUSTERS = 30000
  integer, parameter :: MEAS_CLUSTERS  = 625000000
  integer, parameter :: SAMPLE_EVERY   = 25000

  !----------------------- State -------------------------
  integer :: spin(L,L)
  real(dp) :: Jx(L,L), Jy(L,L)

  !----------------------- Variables ---------------------
  integer :: rid, step, sample_idx
  real(dp) :: T, beta
  character(len=32) :: arg

  !----------------------- Paths -------------------------
  character(len=512) :: outRoot
  character(len=512) :: topDir, deltaDir, realDir, tDir
  character(len=512) :: jxFile, jyFile, obsFile, corrFile, metaFile, outFile

  !----------------------- Measurements ------------------
  real(dp) :: energy_tot
  integer  :: mag_signed
  real(dp) :: m_signed, m_abs, m2
  integer  :: nsamp
  real(dp) :: sum_m2, sum_mabs, sum_m_signed
  real(dp) :: chi
  real(dp) :: sumG(RMAX), avgG(RMAX), Ccorr(RMAX), Gtmp(RMAX)

  ! Per-configuration counts (16 patterns, +/- center)
  integer :: nplus(0:15), nminus(0:15)

  integer :: uObs, uCorr, uMeta, uOut

  ! *** NEW: loop indices for writing bonds into meta.txt ***
  integer :: i, j

  !----------------------- Read realization ID --------------------------
  call get_command_argument(1, arg)
  if (len_trim(arg) == 0) then
     print *, 'ERROR: Provide realisation id'
     stop
  end if
  read(arg,*) rid

  call seed_rng()

  !----------------------- Directories -------------------
  outRoot = './output'
  call mkdir_p(trim(outRoot))

  call make_top_dir(L, topDir)
  call make_delta_dir(delta, deltaDir)
  call make_real_dir(rid, realDir)

  realDir = trim(outRoot)//'/'//trim(topDir)//'/'//trim(deltaDir)//'/'//trim(realDir)
  call mkdir_p(trim(realDir))

  !----------------------- Read quenched disorder ----    ! *** CHANGED ***
  jxFile = trim(realDir)//'/Jx.dat'
  jyFile = trim(realDir)//'/Jy.dat'

  call read_matrix_ascii(jxFile, Jx, L, 'Jx')   ! *** NEW ***
  call read_matrix_ascii(jyFile, Jy, L, 'Jy')   ! *** NEW ***

  !----------------------- Temperature sweep -------------
  T = T_min
  do while (T <= T_max + 0.5_dp*dT)

     beta = 1.0_dp / T

     call make_T_dir(T, tDir)
     call mkdir_p(trim(realDir)//'/'//trim(tDir))

     obsFile  = trim(realDir)//'/'//trim(tDir)//'/obs.dat'
     corrFile = trim(realDir)//'/'//trim(tDir)//'/correlation.dat'
     metaFile = trim(realDir)//'/'//trim(tDir)//'/meta.txt'
     outFile  = trim(realDir)//'/'//trim(tDir)//'/output.dat'

     call init_spins_random(spin, L)

     ! Equilibration
     do step = 1, THERM_CLUSTERS
        call wolff_cluster_flip(spin, Jx, Jy, L, beta)
     end do

     nsamp = 0
     sum_m2 = 0.0_dp
     sum_mabs = 0.0_dp
     sum_m_signed = 0.0_dp
     sumG(:) = 0.0_dp
     sample_idx = 0

     ! Obs in text
     open(newunit=uObs, file=obsFile, status='replace', action='write')
     write(uObs,'(A,F10.5)') '# T = ', T
     write(uObs,'(A)') '# step  E  m_signed  m_abs  m2'

     ! Per-configuration output (RAW ASCII: 32 integers per line)
     open(newunit=uOut, file=outFile, status='replace', action='write')

     do step = 1, MEAS_CLUSTERS
        call wolff_cluster_flip(spin, Jx, Jy, L, beta)

        if (mod(step, SAMPLE_EVERY) == 0) then
           sample_idx = sample_idx + 1

           energy_tot = total_energy_disorder(spin, Jx, Jy, L)
           mag_signed = total_magnetization_int(spin, L)

           m_signed = real(mag_signed,dp) / real(N,dp)
           m_abs    = abs(m_signed)
           m2       = m_signed*m_signed

           write(uObs,'(I12,1X,F14.6,1X,F12.6,1X,F12.6,1X,F12.6)') &
                step, energy_tot, m_signed, m_abs, m2

           nsamp = nsamp + 1
           sum_m2 = sum_m2 + m2
           sum_mabs = sum_mabs + m_abs
           sum_m_signed = sum_m_signed + m_signed

           call spatial_G_of_r(spin, L, RMAX, Gtmp)
           sumG(:) = sumG(:) + Gtmp(:)

           call compute_neighborhood_counts(spin, L, nplus, nminus)
           call write_counts_line(uOut, nplus, nminus)
        end if
     end do

     close(uObs)
     close(uOut)

     ! Write metadata once per (rid, T)
     open(newunit=uMeta, file=metaFile, status='replace', action='write')
     write(uMeta,'(A,I0)')   'L = ', L
     write(uMeta,'(A,I0)')   'N = ', N
     write(uMeta,'(A,F12.6)')'T = ', T
     write(uMeta,'(A,I0)')   'nsamp = ', nsamp
     write(uMeta,'(A,I0)')   'sample_every = ', SAMPLE_EVERY
     write(uMeta,'(A,I0)')   'therm_clusters = ', THERM_CLUSTERS
     write(uMeta,'(A,I0)')   'meas_clusters = ', MEAS_CLUSTERS
     write(uMeta,'(A,A)')    'learning_output = ', 'output.dat'
     write(uMeta,'(A,A)')    'learning_definition = ', &
          'RAW per-config counts: nplus(0) nminus(0) ... nplus(15) nminus(15)'
     write(uMeta,'(A,A)')    'pid_encoding = ', 'pid=8*up+4*down+2*left+right with -1->0,+1->1'

     ! *** NEW: append actual Jx and Jy bond values at the end of meta.txt ***
     write(uMeta,'(A)') 'Jx bonds (LxL):'
     do i = 1, L
        write(uMeta,'(*(F12.6,1X))') (Jx(i,j), j=1,L)
     end do
     write(uMeta,'(A)') 'Jy bonds (LxL):'
     do i = 1, L
        write(uMeta,'(*(F12.6,1X))') (Jy(i,j), j=1,L)
     end do
     ! *** END NEW BLOCK ***

     close(uMeta)

     ! Final stats
     if (nsamp > 0) then
        sum_m2 = sum_m2 / real(nsamp, dp)
        sum_mabs = sum_mabs / real(nsamp, dp)
        chi = beta * real(N,dp) * (sum_m2 - sum_mabs*sum_mabs)

        avgG(:) = sumG(:) / real(nsamp, dp)
        Ccorr(:) = avgG(:) - (sum_m_signed/real(nsamp,dp))**2
     else
        chi = 0.0_dp
        Ccorr(:) = 0.0_dp
     end if

     open(newunit=uCorr, file=corrFile, status='replace', action='write')
     do step = 1, RMAX
        write(uCorr,'(I6,1X,F16.8)') step, Ccorr(step)
     end do
     close(uCorr)

     print '(A,I4,A,F6.3,A,F10.6,A,I0)', 'rid=', rid, ' T=', T, ' chi=', chi, ' nsamp=', nsamp

     T = T + dT
  end do

  print *, 'DONE: realisation ', rid

contains

!------------------------ Directory / naming helpers ------------------------

  subroutine make_top_dir(L, name)
    integer, intent(in) :: L
    character(len=*), intent(out) :: name
    write(name,'("System_Size_L",I0)') L
  end subroutine make_top_dir

  subroutine make_delta_dir(delta, name)
    real(dp), intent(in) :: delta
    character(len=*), intent(out) :: name
    integer :: di
    di = nint(delta*1000.0_dp)
    write(name,'("delta_",I0,".",I3.3)') di/1000, mod(di,1000)
  end subroutine make_delta_dir

  subroutine make_real_dir(rid, name)
    integer, intent(in) :: rid
    character(len=*), intent(out) :: name
    write(name,'("realisation_",I4.4)') rid
  end subroutine make_real_dir

  subroutine make_T_dir(T, name)
    real(dp), intent(in) :: T
    character(len=*), intent(out) :: name
    integer :: ti
    ti = nint(T*1000.0_dp)
    write(name,'("T_",I0,".",I3.3)') ti/1000, mod(ti,1000)
  end subroutine make_T_dir

  subroutine mkdir_p(path)
    character(len=*), intent(in) :: path
    character(len=1200) :: cmd
    cmd = 'mkdir -p "'//trim(path)//'"'
    call execute_command_line(cmd)
  end subroutine mkdir_p

!------------------------ IO helpers ------------------------

  ! *** NEW: reader for Jx.dat / Jy.dat ***
  subroutine read_matrix_ascii(filename, A, L_expected, tag)
    character(len=*), intent(in) :: filename
    real(dp), intent(out) :: A(L_expected,L_expected)
    integer, intent(in) :: L_expected
    character(len=*), intent(in) :: tag
    integer :: u, i, j, L_file
    character(len=256) :: line
    integer :: ios

    open(newunit=u, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
       print *, 'ERROR: cannot open ', trim(filename)
       stop
    end if

    ! Skip first line "# tag"
    read(u,'(A)', iostat=ios) line
    ! Second line "# L = <L>"
    read(u,'(A)', iostat=ios) line
    ! Third line: L
    read(u,*, iostat=ios) L_file
    if (ios /= 0) then
       print *, 'ERROR: failed reading L from ', trim(filename)
       stop
    end if
    if (L_file /= L_expected) then
       print *, 'ERROR: L mismatch in ', trim(filename), &
                ' got ', L_file, ' expected ', L_expected
       stop
    end if

    do i = 1, L_expected
       read(u,*, iostat=ios) (A(i,j), j=1,L_expected)
       if (ios /= 0) then
          print *, 'ERROR: reading row ', i, ' from ', trim(filename)
          stop
       end if
    end do

    close(u)
  end subroutine read_matrix_ascii

!------------------------ Initialization ------------------------

  subroutine init_spins_random(spin, L)
    integer, intent(in) :: L
    integer, intent(out) :: spin(L,L)
    integer :: i, j
    real(dp) :: r
    do i = 1, L
       do j = 1, L
          call random_number(r)
          if (r < 0.5_dp) then
             spin(i,j) = -1
          else
             spin(i,j) = 1
          end if
       end do
    end do
  end subroutine init_spins_random

  ! init_bonds_uniform is now UNUSED but kept for reference
  subroutine init_bonds_uniform(Jx, Jy, L, J0, delta)
    integer, intent(in) :: L
    real(dp), intent(in) :: J0, delta
    real(dp), intent(out) :: Jx(L,L), Jy(L,L)
    integer :: i, j
    real(dp) :: r
    do i = 1, L
       do j = 1, L
          call random_number(r)
          Jx(i,j) = (J0 - delta) + (2.0_dp*delta)*r
          call random_number(r)
          Jy(i,j) = (J0 - delta) + (2.0_dp*delta)*r
       end do
    end do
  end subroutine init_bonds_uniform

!------------------------ Wolff cluster flip ------------------------

  subroutine wolff_cluster_flip(spin, Jx, Jy, L, beta)
    integer, intent(in) :: L
    integer, intent(inout) :: spin(L,L)
    real(dp), intent(in) :: Jx(L,L), Jy(L,L), beta

    logical :: in_cluster(L,L)
    integer :: stack_x(N), stack_y(N)
    integer :: top
    integer :: sx, sy, x, y, xn, yn
    integer :: s0
    real(dp) :: r, Jij, p

    call random_number(r); sx = 1 + int(r*L)
    call random_number(r); sy = 1 + int(r*L)

    in_cluster(:,:) = .false.
    s0 = spin(sx,sy)

    top = 1
    stack_x(1) = sx
    stack_y(1) = sy
    in_cluster(sx,sy) = .true.

    do while (top > 0)
       x = stack_x(top)
       y = stack_y(top)
       top = top - 1

       ! ---- right
       xn = mod(x, L) + 1
       yn = y
       if (.not. in_cluster(xn,yn) .and. spin(xn,yn) == s0) then
          Jij = Jx(x,y)
          p = 1.0_dp - exp(-2.0_dp*beta*Jij)
          call random_number(r)
          if (r < p) then
             top = top + 1
             stack_x(top) = xn
             stack_y(top) = yn
             in_cluster(xn,yn) = .true.
          end if
       end if

       ! ---- left
       xn = mod(x-2+L, L) + 1
       yn = y
       if (.not. in_cluster(xn,yn) .and. spin(xn,yn) == s0) then
          Jij = Jx(xn,y)
          p = 1.0_dp - exp(-2.0_dp*beta*Jij)
          call random_number(r)
          if (r < p) then
             top = top + 1
             stack_x(top) = xn
             stack_y(top) = yn
             in_cluster(xn,yn) = .true.
          end if
       end if

       ! ---- up
       xn = x
       yn = mod(y, L) + 1
       if (.not. in_cluster(xn,yn) .and. spin(xn,yn) == s0) then
          Jij = Jy(x,y)
          p = 1.0_dp - exp(-2.0_dp*beta*Jij)
          call random_number(r)
          if (r < p) then
             top = top + 1
             stack_x(top) = xn
             stack_y(top) = yn
             in_cluster(xn,yn) = .true.
          end if
       end if

       ! ---- down
       xn = x
       yn = mod(y-2+L, L) + 1
       if (.not. in_cluster(xn,yn) .and. spin(xn,yn) == s0) then
          Jij = Jy(x,yn)
          p = 1.0_dp - exp(-2.0_dp*beta*Jij)
          call random_number(r)
          if (r < p) then
             top = top + 1
             stack_x(top) = xn
             stack_y(top) = yn
             in_cluster(xn,yn) = .true.
          end if
       end if
    end do

    do x = 1, L
       do y = 1, L
          if (in_cluster(x,y)) spin(x,y) = -spin(x,y)
       end do
    end do
  end subroutine wolff_cluster_flip

!------------------------ Energy / Magnetization ------------------------

  real(dp) function total_energy_disorder(spin, Jx, Jy, L) result(E)
    integer, intent(in) :: L
    integer, intent(in) :: spin(L,L)
    real(dp), intent(in) :: Jx(L,L), Jy(L,L)
    integer :: x, y, xp, yp
    E = 0.0_dp
    do x = 1, L
       do y = 1, L
          xp = mod(x, L) + 1
          yp = mod(y, L) + 1
          E = E - Jx(x,y)*real(spin(x,y)*spin(xp,y), dp) &
                - Jy(x,y)*real(spin(x,y)*spin(x,yp), dp)
       end do
    end do
  end function total_energy_disorder

  integer function total_magnetization_int(spin, L) result(M)
    integer, intent(in) :: L
    integer, intent(in) :: spin(L,L)
    integer :: x, y
    M = 0
    do x = 1, L
       do y = 1, L
          M = M + spin(x,y)
       end do
    end do
  end function total_magnetization_int

!------------------------ Spatial correlation G(r) ------------------------

  subroutine spatial_G_of_r(spin, L, RMAX, Gout)
    integer, intent(in) :: L, RMAX
    integer, intent(in) :: spin(L,L)
    real(dp), intent(out) :: Gout(RMAX)

    integer :: r, x, y, xr, yr
    integer :: sum_pair
    real(dp) :: norm

    norm = 1.0_dp / (2.0_dp * real(L*L, dp))

    do r = 1, RMAX
       sum_pair = 0
       do x = 1, L
          xr = mod(x-1 + r, L) + 1
          do y = 1, L
             yr = mod(y-1 + r, L) + 1
             sum_pair = sum_pair + spin(x,y)*spin(xr,y) + spin(x,y)*spin(x,yr)
          end do
       end do
       Gout(r) = real(sum_pair, dp) * norm
    end do
  end subroutine spatial_G_of_r

!------------------------ Per-configuration neighborhood counts ------------------------

  subroutine compute_neighborhood_counts(spin, L, nplus, nminus)
    integer, intent(in) :: L
    integer, intent(in) :: spin(L,L)
    integer, intent(out) :: nplus(0:15), nminus(0:15)

    integer :: x, y, xp, xm, yp, ym
    integer :: up, down, left, right, pid
    integer :: upb, downb, leftb, rightb
    integer :: center

    nplus(:) = 0
    nminus(:) = 0

    do x = 1, L
       xp = mod(x, L) + 1
       xm = mod(x-2+L, L) + 1
       do y = 1, L
          yp = mod(y, L) + 1
          ym = mod(y-2+L, L) + 1

          center = spin(x,y)
          up     = spin(x,yp)
          down   = spin(x,ym)
          left   = spin(xm,y)
          right  = spin(xp,y)

          upb    = spin_to_bit(up)
          downb  = spin_to_bit(down)
          leftb  = spin_to_bit(left)
          rightb = spin_to_bit(right)

          pid = 8*upb + 4*downb + 2*leftb + rightb

          if (center == 1) then
             nplus(pid) = nplus(pid) + 1
          else
             nminus(pid) = nminus(pid) + 1
          end if
       end do
    end do
  end subroutine compute_neighborhood_counts

  integer function spin_to_bit(s) result(b)
    integer, intent(in) :: s
    if (s == 1) then
       b = 1
    else
       b = 0
    end if
  end function spin_to_bit

  subroutine write_counts_line(u, nplus, nminus)
    integer, intent(in) :: u
    integer, intent(in) :: nplus(0:15), nminus(0:15)
    integer :: vals(32)
    integer :: p, k

    k = 0
    do p = 0, 15
       k = k + 1
       vals(k) = nplus(p)
       k = k + 1
       vals(k) = nminus(p)
    end do

    write(u,'(*(I0,1X))') (vals(k), k=1,32)
  end subroutine write_counts_line

!------------------------ RNG seeding ------------------------

  subroutine seed_rng()
    integer :: n, k, clock
    integer, allocatable :: seed(:)
    call random_seed(size=n)
    allocate(seed(n))
    call system_clock(count=clock)
    do k = 1, n
       seed(k) = clock + 104729*k
    end do
    call random_seed(put=seed)
    deallocate(seed)
  end subroutine seed_rng

end program wolff_random_bond_ising
