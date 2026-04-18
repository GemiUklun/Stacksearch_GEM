import os,glob,subprocess,datetime, sys, time, re
from optparse import OptionParser
from sys import argv
import numpy as np
import pandas as pd
from presto import rfifind, sifting, psr_utils, psrfits, filterbank, parfile    #From PRESTO
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
from matplotlib import gridspec
from itertools import cycle


def update_inf_name(inffile):
    """Read .inf file and replace the last . with a _ in the name."""
    
    with open(inffile, 'r') as f:
        lines = f.readlines()
    
    # Update the names line
    for i, line in enumerate(lines):
        if line.strip().startswith('Data file name without suffix'):
            last_dot_index = line.rfind(".")
            if last_dot_index != -1:
                lines[i] = line[:last_dot_index] + '_' + line[last_dot_index + 1:]
            break
    
    # Write to output file
    with open(inffile, 'w') as f:
        f.writelines(lines)


class Observation(object):
        def __init__(self, file_name, data_type="psrfits", verbosity_level=1):
                self.file_abspath = os.path.abspath(file_name)
                self.file_nameonly = self.file_abspath.split("/")[-1]
                self.file_basename, self.file_extension = os.path.splitext(self.file_nameonly)
                self.file_buffer_copy = ""

                if data_type =="filterbank":

                        try:
                                object_file = filterbank.FilterbankFile(self.file_abspath)

                                self.N_samples = object_file.nspec
                                self.t_samp_s = object_file.dt
                                self.T_obs_s = self.N_samples * self.t_samp_s
                                self.nbits = object_file.header['nbits']
                                self.nchan = object_file.nchan
                                self.chanbw_MHz = object_file.header['foff']
                                self.bw_MHz = self.nchan * self.chanbw_MHz
                                self.freq_central_MHz = object_file.header['fch1'] + object_file.header['foff']*0.5*object_file.nchan
                                self.freq_high_MHz = np.amax(object_file.freqs)
                                self.freq_low_MHz = np.amin(object_file.freqs)
                                self.MJD_int = int(object_file.header['tstart'])
                                self.Tstart_MJD = object_file.header['tstart']

                                self.source_name = object_file.header['source_name'].strip()

                        except ValueError:
                                if verbosity_level >= 1:
                                        print("WARNING: I got a Value Error! Likely your filterbank data is not 8-,16- or 32-bit. Using PRESTO's 'readfile' to get the necessary information...")

                                try:
                                        self.N_samples        = np.float64(get_command_output_with_pipe("readfile %s" % (self.file_abspath), "grep Spectra").split("=")[-1])                                        
                                        self.t_samp_s         = 1.0e-6*int(get_command_output_with_pipe("readfile %s" % (file_name), "grep Sample").split("=")[-1])
                                        self.T_obs_s          = self.N_samples*self.t_samp_s
                                        self.nbits            = int(get_command_output_with_pipe("readfile %s" % (file_name), "grep bits").split("=")[-1])
                                        self.nchan            = int(get_command_output_with_pipe("readfile %s" % (file_name), "grep channels").split("=")[-1])
                                        self.chanbw_MHz       = np.float64(get_command_output_with_pipe("readfile %s" % (file_name), "grep Channel").split("=")[-1])
                                        self.bw_MHz = self.chanbw_MHz*self.nchan                                        
                                        self.Tstart_MJD       = np.float64(get_command_output_with_pipe("readfile %s" % (file_name), "grep MJD").split("=")[-1])
                                        self.freq_high_MHz    = np.float64(get_command_output_with_pipe("readfile %s" % (file_name), "grep High").split("=")[-1])
                                        self.freq_low_MHz     = np.float64(get_command_output_with_pipe("readfile %s" % (file_name), "grep Low").split("=")[-1])
                                        self.freq_central_MHz = (self.freq_high_MHz + self.freq_low_MHz)/2.0
                                        print(self.N_samples, self.t_samp_s, self.T_obs_s, self.nbits, self.nchan, self.chanbw_MHz, self.bw_MHz, self.Tstart_MJD, self.freq_high_MHz, self.freq_central_MHz, self.freq_low_MHz)
                                except:
                                        print("WARNING: 'readfile' failed. Trying to use 'header' to get the necessary information...")
                                        
                                        self.N_samples        = np.abs(int(get_command_output("header %s -nsamples" % (self.file_abspath)).split()[-1]  ))
                                        self.t_samp_s         = np.float64(      get_command_output("header %s -tsamp"    % (self.file_abspath)).split()[-1]) * 1.0e-6
                                        self.T_obs_s          = np.float64(get_command_output("header %s -tobs"     % (self.file_abspath)).split()[-1])
                                        self.nbits            = int(get_command_output("header %s -nbits"    % (self.file_abspath)).split()[-1])
                                        self.nchan            = int(get_command_output("header %s -nchans"   % (self.file_abspath)).split()[-1])
                                        self.chanbw_MHz       = np.fabs(np.float64(get_command_output("header %s -foff"     % (self.file_abspath)).split()[-1]))
                                        self.bw_MHz = self.chanbw_MHz*self.nchan
                                        self.backend = get_command_output("header %s -machine" % (self.file_abspath)).split()[-1]
                                        self.Tstart_MJD              = np.float64(get_command_output("header %s -tstart"   % (self.file_abspath)).split()[-1])
                                        self.freq_high_MHz    = np.float64(get_command_output("header %s -fch1"     % (self.file_abspath)).split()[-1]) + 0.5*self.chanbw_MHz
                                        self.freq_central_MHz = self.freq_high_MHz - 0.5*self.bw_MHz
                                        self.freq_low_MHz = self.freq_high_MHz - self.bw_MHz
                                        
                                        print(self.N_samples, self.t_samp_s, self.T_obs_s, self.nbits, self.nchan, self.chanbw_MHz, self.bw_MHz, self.backend, self.Tstart_MJD, self.freq_high_MHz, self.freq_central_MHz, self.freq_low_MHz)


                if data_type =="psrfits":
                        if verbosity_level >= 2:
                                print("Reading PSRFITS....")
                        if psrfits.is_PSRFITS(file_name) == True:
                                if verbosity_level >= 2:
                                        print("File '%s' correctly recognized as PSRFITS" % (file_name))
                                object_file = psrfits.PsrfitsFile(self.file_abspath)
                                self.bw_MHz = object_file.specinfo.BW
                                self.N_samples = object_file.specinfo.N
                                self.T_obs_s = object_file.specinfo.T
                                self.backend = object_file.specinfo.backend
                                self.nbits = object_file.specinfo.bits_per_sample
                                self.date_obs = object_file.specinfo.date_obs
                                self.dec_deg = object_file.specinfo.dec2000
                                self.dec_str = object_file.specinfo.dec_str
                                self.chanbw_MHz = object_file.specinfo.df
                                self.t_samp_s = object_file.specinfo.dt
                                self.freq_central_MHz = object_file.specinfo.fctr
                                self.receiver = object_file.specinfo.frontend
                                self.freq_high_MHz = object_file.specinfo.hi_freq
                                self.freq_low_MHz = object_file.specinfo.lo_freq
                                self.MJD_int = object_file.specinfo.mjd
                                self.MJD_sec = object_file.specinfo.secs
                                self.Tstart_MJD = self.MJD_int + np.float64(self.MJD_sec/86400.)
                                self.nchan = object_file.specinfo.num_channels
                                self.observer = object_file.specinfo.observer
                                self.project = object_file.specinfo.project_id
                                self.ra_deg = object_file.specinfo.ra2000
                                self.ra_str = object_file.specinfo.ra_str
                                self.seconds_of_day = object_file.specinfo.secs
                                self.source_name = object_file.specinfo.source
                                self.telescope = object_file.specinfo.telescope

                        else:
                                print("Reading PSRFITS (header only)....")
                                self.bw_MHz = np.float64(get_command_output("vap -n -c bw %s" % (file_name)).split()[-1])
                                self.N_samples = np.float64(get_command_output_with_pipe("readfile %s" % (file_name), "grep Spectra").split("=")[-1])
                                self.T_obs_s = np.float64(get_command_output("vap -n -c length %s" % (file_name)).split()[-1])
                                self.backend = get_command_output("vap -n -c backend %s" % (file_name)).split()[-1]
                                self.nbits = int(get_command_output_with_pipe("readfile %s" % (file_name), "grep bits").split("=")[-1])
                                self.chanbw_MHz = np.float64(get_command_output_with_pipe("readfile %s" % (file_name), "grep Channel").split("=")[-1])
                                self.t_samp_s = np.float64(get_command_output("vap -n -c tsamp %s" % (file_name)).split()[-1])
                                self.freq_central_MHz = np.float64(get_command_output("vap -n -c freq %s" % (file_name)).split()[-1])
                                self.receiver = get_command_output("vap -n -c rcvr %s" % (file_name)).split()[-1]
                                self.freq_high_MHz = np.float64(get_command_output_with_pipe("readfile %s" % (file_name), "grep High").split("=")[-1])
                                self.freq_low_MHz = np.float64(get_command_output_with_pipe("readfile %s" % (file_name), "grep Low").split("=")[-1])
                                self.nchan = int(get_command_output("vap -n -c nchan %s" % (file_name)).split()[-1])
                                self.MJD_int = int(get_command_output("psrstat -Q -c ext:stt_imjd %s" % (file_name)).split()[-1])
                                self.MJD_sec_int = int(get_command_output("psrstat -Q -c ext:stt_smjd %s" % (file_name)).split()[-1])
                                self.MJD_sec_frac = np.float64(get_command_output("psrstat -Q -c ext:stt_offs %s" % (file_name)).split()[-1])
                                self.MJD_sec = self.MJD_sec_int + self.MJD_sec_frac
                                self.Tstart_MJD       = self.MJD_int + np.float64(self.MJD_sec/86400.)


class Pulsar(object):
        def __init__(self, parfilename):
                LIGHT_SPEED = 2.99792458e10   # Light speed in CGS units

                pulsar_parfile = parfile.psr_par(parfilename)
                
                self.parfilename = parfilename
                if hasattr(pulsar_parfile, 'PSR'):
                        self.psr_name = pulsar_parfile.PSR
                elif hasattr(pulsar_parfile, 'PSRJ'):
                        self.psr_name = pulsar_parfile.PSRJ

                self.PEPOCH = pulsar_parfile.PEPOCH
                self.F0 = pulsar_parfile.F0
                self.P0_s = 1./self.F0
                self.P0_ms = self.P0_s * 1000
                if hasattr(pulsar_parfile, 'F1'):
                        self.F1 = pulsar_parfile.F1
                else:
                        self.F1 = 0
                if hasattr(pulsar_parfile, 'F2'):
                        self.F2 = pulsar_parfile.F2
                else:
                        self.F2 = 0

                self.is_binary = hasattr(pulsar_parfile, 'BINARY')

                if self.is_binary:
                        self.pulsar_type = "binary"
                        self.binary_model = pulsar_parfile.BINARY

                        # 1) Orbital period
                        if hasattr(pulsar_parfile, 'PB'):
                                self.Pb_d = pulsar_parfile.PB
                                self.Pb_s = self.Pb_d*86400
                                self.Fb0 = 1./self.Pb_s
                        elif hasattr(pulsar_parfile, 'FB0'):
                                self.Fb0 = pulsar_parfile.FB0
                                self.Pb_s = 1./self.Fb0
                                self.Pb_d = self.Pb_s / 86400.

                        # 2) Projected semi-major axis of the pulsar orbit
                        self.x_p_lts = pulsar_parfile.A1
                        self.x_p_cm = pulsar_parfile.A1 * LIGHT_SPEED

                        # 3) Orbital eccentricity
                        if hasattr(pulsar_parfile, 'E'):
                                self.ecc = pulsar_parfile.E
                        elif hasattr(pulsar_parfile, 'ECC'):
                                self.ecc = pulsar_parfile.ECC
                        elif hasattr(pulsar_parfile, 'EPS1') and hasattr(pulsar_parfile, 'EPS2'):
                                self.eps1 = pulsar_parfile.EPS1
                                self.eps2 = pulsar_parfile.EPS2
                                self.ecc = np.sqrt(self.eps1**2 + self.eps2**2)
                        else:
                                self.ecc = 0

                        # 4) Longitude of periastron
                        if hasattr(pulsar_parfile, 'OM'):
                                self.omega_p_deg = pulsar_parfile.OM
                        else:
                                self.omega_p_deg = 0
                        self.omega_p_rad = self.omega_p_deg * np.pi/180

                        # 5) Epoch of passage at periastron/ascending node
                        if hasattr(pulsar_parfile, 'T0'):
                                self.T0 = pulsar_parfile.T0
                                self.Tasc = self.T0
                        elif hasattr(pulsar_parfile, 'TASC'):
                                self.Tasc = pulsar_parfile.TASC
                                self.T0 = self.Tasc

                        self.v_los_max = (2*np.pi * self.x_p_cm / self.Pb_s)
                        self.doppler_factor = self.v_los_max / LIGHT_SPEED

                else:
                        # If the pulsar is isolated
                        self.pulsar_type = "isolated"
                        self.v_los_max = 0
                        self.doppler_factor = 1e-4  # Account for Doppler shift due to the Earth motion around the Sun

def get_command_output(command, shell_state=False, work_dir=os.getcwd()):
        list_for_Popen = command.split()
        
        #print "alex_processes::get_command_output: %s" % (command)                                                       
        #print list_for_Popen
        #print "work_dir=%s" % (work_dir)
        if shell_state==False:
                proc = subprocess.Popen(list_for_Popen, stdout=subprocess.PIPE, shell=shell_state, cwd=work_dir)
        else:
                proc = subprocess.Popen([command], stdout=subprocess.PIPE, shell=shell_state, cwd=work_dir)
        out, err = proc.communicate()
        #print "OUT: '%s'" % (out)                                                                                          
        return out.decode('utf-8')

def get_command_output_with_pipe(command1, command2):

    list_for_Popen_cmd1 = command1.split()
    list_for_Popen_cmd2 = command2.split()

    p1 = subprocess.Popen(list_for_Popen_cmd1, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(list_for_Popen_cmd2, stdin=p1.stdout, stdout=subprocess.PIPE)
    p1.stdout.close() 

    out, err = p2.communicate()
    return out.decode('utf-8')

def get_DDplan_scheme(infile, out_dir, LOG_basename, loDM, highDM, DM_coherent_dedispersion, freq_central_MHz, bw_MHz, nchan, nsubbands, t_samp_s):
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]
        log_abspath = "%s/LOG_%s.txt" % (os.getcwd(), LOG_basename)

        print("DM_coherent_dedispersion == ", DM_coherent_dedispersion)
        if DM_coherent_dedispersion == 0:
                cmd_DDplan = "DDplan.py -o ddplan_%s -l %s -d %s -f %s -b %s -n %s -t %s" % (infile_basename, loDM, highDM, freq_central_MHz, np.fabs(bw_MHz), nchan, t_samp_s)
        elif DM_coherent_dedispersion > 0:
                cmd_DDplan = "DDplan.py -o ddplan_%s -l %s -d %s -c %s -f %s -b %s -n %s -t %s" % (infile_basename, loDM, highDM, DM_coherent_dedispersion, freq_central_MHz, np.fabs(bw_MHz), nchan, t_samp_s)
                print("Coherent dedispersion enabled. Command would be: ", cmd_DDplan)
        elif DM_coherent_dedispersion < 0:
                print("ERROR: The DM of coherent dedispersion < 0! Exiting...")
                exit()

        print("Running: %s " % (cmd_DDplan))
        output_DDplan    = get_command_output(cmd_DDplan, shell_state=False, work_dir=out_dir)

        list_DD_schemes  = get_DD_scheme_from_DDplan_output(output_DDplan)


        return list_DD_schemes
                

def get_DD_scheme_from_DDplan_output(output_DDplan):
        list_dict_schemes = []

        print(output_DDplan)
        output_DDplan_list_lines = output_DDplan.split("\n")
        index = output_DDplan_list_lines.index("  Low DM    High DM     dDM  DownSamp   #DMs  WorkFract")   +1
        print(output_DDplan)
        flag_add_plans = 1
        while flag_add_plans == 1:
                if output_DDplan_list_lines[index] == "":
                        return list_dict_schemes
                else:
                        param = output_DDplan_list_lines[index].split()
                        low_DM      = float(param[0])
                        high_DM     = float(param[1])
                        dDM         = float(param[2])
                        downsamp    = int(param[3])
                        num_DMs     = int(param[4])

                        if num_DMs > 1000 and num_DMs <= 2000:
                                diff_num_DMs = num_DMs - 1000
                                low_DM1   = low_DM
                                high_DM1  = low_DM + 1000*dDM

                                low_DM2   = high_DM1
                                high_DM2  = high_DM

                                dict_scheme1 = {'loDM': low_DM1, 'highDM': high_DM1, 'dDM': dDM, 'downsamp': downsamp, 'num_DMs': 1000 }
                                dict_scheme2 = {'loDM': low_DM2, 'highDM': high_DM2, 'dDM': dDM, 'downsamp': downsamp, 'num_DMs': diff_num_DMs }
                                list_dict_schemes.append(dict_scheme1)
                                list_dict_schemes.append(dict_scheme2)
                        else:
                                dict_scheme = {'loDM': low_DM, 'highDM': high_DM, 'dDM': dDM, 'downsamp': downsamp, 'num_DMs': num_DMs }
                                list_dict_schemes.append(dict_scheme)

                index = index + 1


def execute_and_log(command, work_dir, log_abspath, dict_envs={}, flag_append=0, verbosity_level=0):

        datetime_start = (datetime.datetime.now()).strftime("%Y/%m/%d  %H:%M")
        time_start = time.time()
        if flag_append == 1:
                flag_open_mode = "a"
        else:
                flag_open_mode = "w+"
        log_file = open("%s" % (log_abspath), flag_open_mode)
        executable = command.split()[0]
        

        log_file.write("****************************************************************\n")
        log_file.write("START DATE AND TIME: %s\n" % (datetime_start))
        log_file.write("\nCOMMAND:\n")
        log_file.write("%s\n\n" % (command))
        log_file.write("WORKING DIRECTORY: %s\n" % (work_dir))
        log_file.write("****************************************************************\n")
        log_file.flush()
        

        list_for_Popen = command.split()
        env_subprocess = os.environ.copy()
        if dict_envs: #If the dictionary is not empty                                                                                                                                                            
                for k in list(dict_envs.keys()):
                        env_subprocess[k] = dict_envs[k]
        
        proc = subprocess.Popen(list_for_Popen, stdout=log_file, stderr=log_file, cwd=work_dir, env=env_subprocess)
        proc.communicate()  #Wait for the process to complete                                                                                                                                                    

        datetime_end = (datetime.datetime.now()).strftime("%Y/%m/%d  %H:%M")
        time_end = time.time()

        if verbosity_level >= 1:
                print("execute_and_log:: COMMAND: %s" % (command))
                print("execute_and_log:: which %s: "% (executable), get_command_output("which %s" % (executable)))
                print("execute_and_log:: WORKING DIRECTORY = ", work_dir)
                print("execute_and_log:: CHECK LOG WITH: \"tail -f %s\"" % (log_abspath)); sys.stdout.flush()
                print("execute_and_log: list_for_Popen = ", list_for_Popen)
                print("execute_and_log: log_file       = ", log_file)
                print("execute_and_log: env_subprocess = ", env_subprocess)

        log_file.write("\nEND DATE AND TIME: %s\n" % (datetime_end))
        log_file.write("\nTOTAL TIME TAKEN: %d s\n" % (time_end - time_start))
        log_file.close()


def prepsubband(infile, out_dir, LOG_filename, segment_label, chunk_label, Nsamples, mask_file, list_DD_schemes, nchan, nsubbands=0, other_flags="", verbosity_level=0, n_cpu=None):
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]
        print("prepsubband:: infile_basename = ", infile_basename)
        datfile_basename = infile_basename
        dict_env = {}
        log_abspath = LOG_filename
        N_schemes = len(list_DD_schemes)

        string_mask = ""
        if mask_file != "":  string_mask = "-mask %s" % (mask_file)

        print("----------------------------------------------------------------------")
        print("prepsubband will be run %d times with the following DM ranges:" % (N_schemes))
        print()
        print("%10s %10s %10s %10s %10s " % ("Low DM", "High DM", "dDM",  "DownSamp",   "#DMs"))
        for i in range(N_schemes):
                print("%10s %10s %10s %10s %10s " % (list_DD_schemes[i]['loDM'], float(list_DD_schemes[i]['loDM']) + int(list_DD_schemes[i]['num_DMs'])*float(list_DD_schemes[i]['dDM']), list_DD_schemes[i]['dDM'] ,  list_DD_schemes[i]['downsamp'],  list_DD_schemes[i]['num_DMs'] ))
        print(); sys.stdout.flush()

        if nsubbands == 0:
                nsubbands = nchan
        elif (nchan % nsubbands != 0):
                print("ERROR: requested number of subbands is %d, which is not an integer divisor of the number of channels %d! " % (nsubbands, nchan))
                exit()

        print("Dedispersing with %d subbands (original number of channels: %d)..." % (nsubbands, nchan))

        if verbosity_level >= 0:
                print("prepsubband::  Checking prepsubband results...")
        if check_prepsubband_result(out_dir, list_DD_schemes, datfile_basename, verbosity_level=2) == True:
                print() 
                print("prepsubband:: WARNING: all the dedispersed time series for %s are already there! Skipping." % (infile_basename))
        
        elif check_prepsubband_result_DMFOLD(out_dir, list_DD_schemes, datfile_basename, verbosity_level=2) == True:
                print() 
                print("prepsubband:: WARNING: all the dedispersed time series for %s are already there! Skipping." % (infile_basename))

        else:
                while check_prepsubband_result(out_dir, list_DD_schemes, datfile_basename, verbosity_level=1) == False:
                        if verbosity_level >= 1:
                                print("check_prepsubband_result(out_dir, list_DD_schemes, verbosity_level=0) = ", check_prepsubband_result(out_dir, list_DD_schemes, datfile_basename, verbosity_level=0))
                                print("Checking results at: %s" % (out_dir))
                        if verbosity_level >= 1:
                                print("\033[1m >> TIP:\033[0m Check prepsubband progress with '\033[1mtail -f %s\033[0m'" % (log_abspath))
                                print()
                        for i in range(N_schemes):
                                if Nsamples > 0:
                                        downsamp_factor = int(list_DD_schemes[i]['downsamp'])
                                        flag_numout = "-numout %d" % (Nsamples/downsamp_factor) 

                                prepsubband_outfilename = "%s_%s_%s" % (infile_basename, segment_label, chunk_label)
                                cmd_prepsubband = "prepsubband -ncpus %d %s %s -o %s %s -lodm %s -dmstep %s -numdms %s -downsamp %s -nsub %s %s" % (n_cpu,other_flags, flag_numout, prepsubband_outfilename, string_mask, list_DD_schemes[i]['loDM'], list_DD_schemes[i]['dDM'], list_DD_schemes[i]['num_DMs'], list_DD_schemes[i]['downsamp'], nsubbands, infile)
                                if infile.endswith(".fits"):
                                       cmd_prepsubband = cmd_prepsubband + " -noscales -nooffsets"
                                print("Running:")
                                print(cmd_prepsubband)
                                print() 
                                print("Running prepsubband with scheme %d/%d on observation '%s'..." % (i+1, N_schemes, infile), end=' ') ; sys.stdout.flush()

                                if verbosity_level >= 1:
                                        print("dedisperse:: %d) RUNNING: %s" % (i, cmd_prepsubband))
                                execute_and_log("which prepsubband", out_dir, log_abspath, dict_env, 1)
                                execute_and_log(cmd_prepsubband, out_dir, log_abspath, dict_env, 1)
                                print("done!"); sys.stdout.flush()

                                # Rename output files to replace dot with underscore in DM values
                                for dm in np.arange(list_DD_schemes[i]['loDM'], list_DD_schemes[i]['highDM'] - 0.5*list_DD_schemes[i]['dDM'], list_DD_schemes[i]['dDM']):
                                    dm_str_dot = "DM%.2f" % dm  # e.g., "DM88.49"
                                    dm_str_underscore = dm_str_dot.replace(".", "_")  # e.g., "DM88_49"
                                    for ext in ['.dat', '.inf']:
                                        old_pattern = os.path.join(out_dir, "%s*%s%s" % (prepsubband_outfilename, dm_str_dot, ext))
                                        for old_file in glob.glob(old_pattern):
                                            new_file = old_file.replace(dm_str_dot, dm_str_underscore)
                                            if old_file != new_file:
                                                os.rename(old_file, new_file)
                                                if verbosity_level >= 1:
                                                    print("Renamed: %s -> %s" % (os.path.basename(old_file), os.path.basename(new_file)))
                                                if new_file.endswith(".inf"):
                                                       update_inf_name(new_file)

def handle_error(e):
    print(f"ERROR in prepsubband: {e}")

def check_prepsubband_result(work_dir, list_DD_schemes, datfile_basename, verbosity_level=1):
        N_schemes = len(list_DD_schemes)
        if verbosity_level >= 1:
                print("check_prepsubband_result:: list_DD_schemes = ", list_DD_schemes)
                print("check_prepsubband_result:: work_dir = ", work_dir)

        for i in range(N_schemes):
                for dm in np.arange(list_DD_schemes[i]['loDM'],   list_DD_schemes[i]['highDM'] - 0.5*list_DD_schemes[i]['dDM']        , list_DD_schemes[i]['dDM']):
                        if verbosity_level >= 1:
                                print("check_prepsubband_result:: Looking for: ", os.path.join(work_dir, "%s*DM%s.dat"%(datfile_basename, ("%.2f" % dm).replace(".","_")) ),  os.path.join(work_dir, "%s*DM%s.inf"%(datfile_basename, ("%.2f" % dm).replace(".","_")) ))
                                print("check_prepsubband_result:: This is what I found: %s, %s" % (  [ x for x in glob.glob(os.path.join(work_dir, "%s*DM%s.dat"%(datfile_basename, ("%.2f" % dm).replace(".","_")))) if not "_red" in x]  , [ x for x in glob.glob(os.path.join(work_dir, "%s*DM%s.inf"%(datfile_basename, ("%.2f" % dm).replace(".","_")))) if not "_red" in x]    ))
                        if len( [ x for x in glob.glob(os.path.join(work_dir, "%s*DM%s.dat"%(datfile_basename, ("%.2f" % dm).replace(".","_")))) if not "_red" in x]   + [ x for x in glob.glob(os.path.join(work_dir, "%s*DM%s.inf"%(datfile_basename, ("%.2f" % dm).replace(".","_")))) if not "_red" in x] ) != 2:
                                if verbosity_level >= 1:
                                        print("check_prepsubband_result: False")
                                return False
        if verbosity_level >= 1:
                print("check_prepsubband_result: True")

        return True

def check_prepsubband_result_DMFOLD(work_dir, list_DD_schemes, datfile_basename, verbosity_level=1):
        N_schemes = len(list_DD_schemes)
         
        if verbosity_level >= 1:
                print("check_prepsubband_result:: list_DD_schemes = ", list_DD_schemes)
                print("check_prepsubband_result:: work_dir = ", work_dir)

        for i in range(N_schemes):
                for dm in np.arange(list_DD_schemes[i]['loDM'],   list_DD_schemes[i]['highDM'] - 0.5*list_DD_schemes[i]['dDM']        , list_DD_schemes[i]['dDM']):
                        OLD_work_dir = work_dir
                        Curr_DM = "/DM%s.dat"%( ("%.2f" % dm).replace(".","_"))
                        Curr_DM = Curr_DM[:-4]
                        work_dir = work_dir + Curr_DM
                        if verbosity_level >= 1:
                                print("check_prepsubband_result:: Looking for: ", os.path.join(work_dir, "%s*DM%s.dat"%(datfile_basename, ("%.2f" % dm).replace(".","_")) ),  os.path.join(work_dir, "%s*DM%s.inf"%(datfile_basename, ("%.2f" % dm).replace(".","_")) ))
                                print("check_prepsubband_result:: This is what I found: %s, %s" % (  [ x for x in glob.glob(os.path.join(work_dir, "%s*DM%s.dat"%(datfile_basename, ("%.2f" % dm).replace(".","_")))) if not "_red" in x]  , [ x for x in glob.glob(os.path.join(work_dir, "%s*DM%s.inf"%(datfile_basename, ("%.2f" % dm).replace(".","_")))) if not "_red" in x]    ))
                        if len( [ x for x in glob.glob(os.path.join(work_dir, "%s*DM%s.dat"%(datfile_basename, ("%.2f" % dm).replace(".","_")))) if not "_red" in x]   + [ x for x in glob.glob(os.path.join(work_dir, "%s*DM%s.inf"%(datfile_basename, ("%.2f" % dm).replace(".","_")))) if not "_red" in x] ) != 2:
                                if verbosity_level >= 1:
                                        print("check_prepsubband_result: False")
                                return False
                        work_dir = OLD_work_dir
        if verbosity_level >= 1:
                print("check_prepsubband_result: True")

        return True

def check_stacksearch_result(work_dir, DM_code, verbosity_level=1):

        Out_code = DM_code + "STACK"
         
        if verbosity_level >= 1:
                print("check_stacksearch_result:: name = ", Out_code)
                print("check_stacksearch_result:: work_dir = ", work_dir)

        if verbosity_level >= 1:
                print("check_stacksearch_result:: Looking for: ", os.path.join(work_dir,Out_code))

        if len(glob.glob(os.path.join(work_dir,Out_code))) == 0:
                if verbosity_level >= 1:
                        print("check_prepsubband_result: False")
                return False
        if verbosity_level >= 1:
                print("check_prepsubband_result: True")

        return True


def realfft(infile, out_dir, LOG_filename, other_flags="", verbosity_level=0, flag_LOG_append=0):
        infile_nameonly = os.path.basename(infile)
        infile_basename = os.path.splitext(infile_nameonly)[0]
        log_abspath = LOG_filename
        fftfile_abspath = os.path.join(out_dir, "%s.fft" % (infile_basename) )
        cmd_realfft = "realfft %s %s" % (other_flags, infile)
        if verbosity_level >= 1:
                print("%s | realfft:: Running:" % (datetime.datetime.now()).strftime("%Y/%m/%d  %H:%M")); sys.stdout.flush()
                print("%s" % (cmd_realfft)) ; sys.stdout.flush()

        if os.path.exists( fftfile_abspath ) and (os.path.getsize(fftfile_abspath) > 0):
                if verbosity_level >= 1:
                        print()
                        print("WARNING: File %s already present. Skipping realfft..." % (fftfile_abspath))
        else:
                dict_env = {}
                
                execute_and_log(cmd_realfft, out_dir, log_abspath, dict_env, 0)
                if os.path.exists( fftfile_abspath ) and (os.stat(fftfile_abspath).st_size > 0):
                        if verbosity_level >= 1:
                                print("%s | realfft on \"%s\" completed successfully!" % (datetime.datetime.now().strftime("%Y/%m/%d  %H:%M"), infile_nameonly)); sys.stdout.flush()
                else:
                        print("WARNING (%s) | could not find all the output files from realfft on \"%s\"!" % (datetime.datetime.now().strftime("%Y/%m/%d  %H:%M"), infile_nameonly)); sys.stdout.flush()


def check_zaplist_outfiles(fft_infile, verbosity_level=0):
        birds_filename    = fft_infile.replace(".fft", ".birds")
        zaplist_filename  = fft_infile.replace(".fft", ".zaplist")
        try:
                if (os.path.getsize(birds_filename) > 0) and (os.path.getsize(zaplist_filename)>0): #checks if it exists and its
                        return True
                else:
                        return False
        except OSError:
                return False


def rednoise(fftfile, out_dir, LOG_filename, other_flags="", verbosity_level=0):
        #print "rednoise:: Inside rednoise"
        fftfile_nameonly = os.path.basename(fftfile)
        fftfile_basename = os.path.splitext(fftfile_nameonly)[0]
        log_abspath = LOG_filename
        

        dereddened_ffts_filename = "%s/dereddened_ffts.txt" % (out_dir)
        fftfile_rednoise_abspath = os.path.join(out_dir, "%s_red.fft" % (fftfile_basename) )
        inffile_original_abspath = os.path.join(out_dir, "%s.inf" % (fftfile_basename) )
        
        
        cmd_rednoise = "rednoise %s %s" % (other_flags, fftfile)


        if verbosity_level >= 1:
                print("rednoise:: dereddened_ffts_filename = ", dereddened_ffts_filename)
                print("rednoise:: fftfile_rednoise_abspath = ", fftfile_rednoise_abspath)
                print("rednoise:: cmd_rednoise = ", cmd_rednoise)
                #print "%s | Running:" % (datetime.datetime.now()).strftime("%Y/%m/%d  %H:%M"); sys.stdout.flush()
                #print "%s" % (cmd_rednoise) ; sys.stdout.flush()
                print("rednoise:: opening '%s'" % (dereddened_ffts_filename))

        try:
                file_dereddened_ffts = open(dereddened_ffts_filename, 'r')
        except:
                if verbosity_level >= 1:           print("rednoise:: File '%s' does not exist. Creating it..." % (dereddened_ffts_filename), end=' ') ; sys.stdout.flush()
                os.mknod(dereddened_ffts_filename)
                if verbosity_level >= 1:           print("done!") ; sys.stdout.flush()
                file_dereddened_ffts = open(dereddened_ffts_filename, 'r')

        # If the fftfile is already in the list of dereddened files...
        if "%s\n" % (fftfile) in file_dereddened_ffts.readlines():
                if verbosity_level >= 1:
                        print("rednoise:: NB: File '%s' is already in the list of dereddened files (%s)." % (fftfile, dereddened_ffts_filename))
                        # Then check is the file has size > 0...
                        print("rednoise:: Checking the size of '%s'" % (fftfile))

                if (os.path.getsize(fftfile) > 0):
                        operation="skip"
                        if verbosity_level >= 1:
                                print("rednoise:: size is > 0. Then skipping...")
                else:
                        operation="make_from_scratch"
                        if verbosity_level >= 1:
                                print("rednoise:: size is = 0. Making from scratch...")

        else:
                operation="make_from_scratch"
                if verbosity_level >= 1:
                        print("rednoise:: File '%s' IS NOT in the list of dereddened files (%s). I will make the file from scratch..." % (fftfile_basename, dereddened_ffts_filename))

                
        file_dereddened_ffts.close()

        if operation=="make_from_scratch":
                if verbosity_level >= 1:
                        print("rednoise:: making the file from scratch...")
                dict_env = {}
                execute_and_log(cmd_rednoise, out_dir, log_abspath, dict_env, 0)
                if verbosity_level >= 1:
                        print("done!", end=' ') ; sys.stdout.flush()
                file_dereddened_ffts = open(dereddened_ffts_filename, 'a')
                file_dereddened_ffts.write("%s\n" % (fftfile))
                file_dereddened_ffts.close()
                # os.rename(fftfile_rednoise_abspath, fftfile_rednoise_abspath.replace("_red.", "."))


def realfft_and_rednoise(datfile, workdir_current_dm, log_filename, VERBOSITY):
    dm_log = log_filename.replace(".log", "_%s.log" % os.path.basename(datfile))  # unique log per file
    
    print("Doing realfft on %s..." % datfile); sys.stdout.flush()
    realfft(datfile, workdir_current_dm, dm_log, "",
            verbosity_level=int(VERBOSITY), flag_LOG_append=1)
    
    fftfile = datfile.replace(".dat", ".fft")
    print("Doing rednoise on %s..." % fftfile); sys.stdout.flush()
    rednoise(fftfile, workdir_current_dm, dm_log, "",
             verbosity_level=int(VERBOSITY))

def stacksearch(fftlist, out_dir, LOG_filename, verbosity_level=0, threshold=8, maxcands=100, nharms=16):
        log_abspath = LOG_filename
        Under_idx = [match.start() for match in re.finditer(r'_', fftlist[0])] # Find all the _ in the first fft file
        DM_code = fftlist[0][Under_idx[-3]+1:Under_idx[-1]+1]
        Out_code = DM_code + "STACK"
        cmd_stacksearch = "stacksearch.py -o %s -t %s -c %s -n %s " % (Out_code, str(threshold), str(maxcands), str(nharms))
        for i in range(len(fftlist)):
                cmd_stacksearch = cmd_stacksearch + fftlist[i] + " "

        if verbosity_level >= 1:
                print("stacksearch:: list of ffts = ", fftlist)
                print("stacksearch:: cmd_stacksearch = ", cmd_stacksearch)

        CHECK_RES = check_stacksearch_result(out_dir, DM_code, verbosity_level)
        if CHECK_RES:
               print("stacksearch result already present!")
        else:
                if verbosity_level >= 1:
                        print("stacksearch:: making the file from scratch...")
                dict_env = {}
                execute_and_log(cmd_stacksearch, out_dir, log_abspath, dict_env, 0)
                if verbosity_level >= 1:
                        print("done!", end=' ') ; sys.stdout.flush()
    

def stacksearch_results_individual(verbosity_level=0, nharms=16,known_pulsars = None, known_freq_low = None, known_freq_high = None, dir_dm = None, REF_fourier_bin = None, STACK_TH=None, BIN_FACTOR=None):

        if VERBOSITY:
               print()
               print("STACKSEARCH INDIVIDUAL SIFTING")
        Results_file = glob.glob("*STACK")[0]
        print(Results_file)
        try:
                # Read the results file, skipping commented lines
                Results = np.loadtxt(Results_file, comments='#')
        except:
                if verbosity_level >= 1:  
                        print("stacksearch_results:: STACK file does not exist.")
                        sys.stdout.flush()
                exit()

        
        sigma = Results[:, 0]
        freq_hz = Results[:, 1]
        period_ms = Results[:, 2]
        fourier_bin = Results[:, 3]
        power = Results[:, 4]
        n_harm = Results[:, 5]

        N_cands = len(sigma)

        Flag_cands = np.ones(N_cands) * -1
        # -1 --> still to analyze
        # 0 --> to remove
        # 1 --> Known pulsar
        # 2 --> Interesting

        Name_cands = np.zeros(N_cands).astype("str") # ALL '0.0' at start
        Counter_interesting = int(0)

        for cand_idx in range(N_cands):
                Curr_num = 0
                Curr_den = 0
                if Flag_cands[cand_idx] == -1:
                        is_known, psrname, Curr_num, Curr_den, known_idx = check_if_cand_is_known(period_ms[cand_idx],known_pulsars,known_freq_low,known_freq_high,cand_idx,nharms)
        
                        if is_known:
                                Name_cands[cand_idx] = psrname
                                Flag_cands[cand_idx] = 1
                                Flag_cands, Name_cands = remove_harmonics(period_ms[cand_idx],period_ms,Flag_cands[cand_idx],Flag_cands,Name_cands,nharms,Curr_num,Curr_den,REF_fourier_bin,BIN_FACTOR,known_freq_low[known_idx],known_freq_high[known_idx],cand_idx) # Putting zeros in the correspondent harmonics
                        
                        else:
                                Name_cands[cand_idx] = dir_dm + "_CAND_" + str(Counter_interesting)
                                Flag_cands[cand_idx] = 2
                                Counter_interesting = Counter_interesting + 1
                                Flag_cands, Name_cands = remove_harmonics(period_ms[cand_idx],period_ms,Flag_cands[cand_idx],Flag_cands,Name_cands,nharms,Curr_num,Curr_den,REF_fourier_bin,BIN_FACTOR,-999,-999,cand_idx) # Putting zeros in the correspondent harmonics


        sigma = sigma[Flag_cands!=0]
        freq_hz = freq_hz[Flag_cands!=0]
        period_ms = period_ms[Flag_cands!=0]
        fourier_bin = fourier_bin[Flag_cands!=0]
        power = power[Flag_cands!=0]
        n_harm = n_harm[Flag_cands!=0]

        Name_cands = Name_cands[Flag_cands!=0]
        Flag_cands = Flag_cands[Flag_cands!=0]

        Curr_individual_results = pd.DataFrame({"Name":Name_cands,
                                   "Code":Flag_cands,
                                   "Sigma":sigma,
                                   "Freq_hz":freq_hz,
                                   "Period_ms":period_ms,
                                   "Fourier_bin":fourier_bin,
                                   "Power":power,
                                   "N_harm":n_harm})

        Curr_individual_results['Sigma'] = Curr_individual_results['Sigma'].fillna(STACK_TH)

        Curr_individual_results = Curr_individual_results.sort_values(by='Sigma', ascending=False)

        Curr_individual_results = Curr_individual_results.reset_index(drop=True)

        Name_out = dir_dm + "_STACK_CAND_INDIVIDUAL.csv"
        Curr_individual_results.to_csv(Name_out,index=False)
        if VERBOSITY:
                pd.set_option('display.width', 1000)
                pd.set_option('display.max_columns', None)  # Show all columns
                pd.set_option('display.max_rows', None)     # Show all rows (optional)

                print(Curr_individual_results)


def stacksearch_results_DMcross(DM_list=None, verbosity_level=0, fourier_bin = None, numharm=None, Min_threshold = None, BIN_FACTOR=None):
        if VERBOSITY:
               print()
               print("STACKSEARCH MULTI-DM SIFTING")

        DM_cross_names = {}
        DM_cross_sigmas = {}
        DM_cross_powers = {}
        DM_cross_periods_ms = {}
        DM_cross_individual_flags = {}
        DM_cross_multi_flags = {}

        if not os.path.exists("RESULTS_DMCROSS"):
                os.mkdir("RESULTS_DMCROSS")


        for dms in DM_list:
               
                Curr_result_file_name = glob.glob("%s/*STACK_CAND_INDIVIDUAL*" % (dms))[0]
                try:
                        # Read the results file
                        Curr_result_file = pd.read_csv(Curr_result_file_name)
                except:
                        if verbosity_level >= 1:  
                                print("stacksearch_results:: Individual STACK file in %s does not exist." % (dms))
                                sys.stdout.flush()
                        exit()

                Curr_names = np.array(Curr_result_file.Name).astype("str")
                Curr_sigmas = np.array(Curr_result_file.Sigma).astype("float")
                Curr_powers = np.array(Curr_result_file.Power).astype("float")
                Curr_periods_ms = np.array(Curr_result_file.Period_ms).astype("float")
                Curr_individual_flags = np.array(Curr_result_file.Code).astype("int")
                Curr_multi_flags = np.ones(len(Curr_names)) * (-1) # -1 --> to be analysed, 0 --> already analysed

                DM_cross_names[dms] = Curr_names
                DM_cross_sigmas[dms] = Curr_sigmas
                DM_cross_powers[dms] = Curr_powers
                DM_cross_periods_ms[dms] = Curr_periods_ms
                DM_cross_individual_flags[dms] = Curr_individual_flags
                DM_cross_multi_flags[dms] = Curr_multi_flags


        Single_candidates_names = []
        Single_candidates_ndms = []
        Single_candidates_maxdms_sigma = []
        Single_candidates_maxdms_power = []
        Single_candidates_sigmas = []
        Single_candidates_powers = []
        Single_candidates_periods_ms = []

        Multi_candidates_names = []
        Multi_candidates_ndms = []
        Multi_candidates_maxdms_sigma = []
        Multi_candidates_maxdms_power = []
        Multi_candidates_sigmas = []
        Multi_candidates_powers = []
        Multi_candidates_periods_ms = []

        Count_single = 0

        Count_Multi = 0



        for REF_dms in DM_list:

                REF_names = DM_cross_names[REF_dms]
                REF_sigmas = DM_cross_sigmas[REF_dms]
                REF_powers = DM_cross_powers[REF_dms]
                REF_periods_ms = DM_cross_periods_ms[REF_dms]
                REF_individual_flags = DM_cross_individual_flags[REF_dms]
                REF_multi_flags = DM_cross_multi_flags[REF_dms]
                REF_N_cands = len(REF_names)

                REF_DM = float(REF_dms.replace("DM", "").replace("_","."))

                for i in range(REF_N_cands):
                       if (REF_multi_flags[i] == -1):
                                
                                Count_DM_cand = 0
                              
                                DM_cross_multi_flags[REF_dms][i] = 0

                                # Plot init
                                Plot_names = []
                                Plot_DMs = []
                                Plot_sigmas = []
                                Plot_powers = []
                                Plot_harms = []

                                # Add the first part of the plot

                                Plot_names.append(REF_names[i])
                                Plot_DMs.append(REF_DM)
                                Plot_harms.append("1/1")
                                Plot_sigmas.append(REF_sigmas[i])
                                Plot_powers.append(REF_powers[i])
                                

                                for COMP_dms in DM_list:
                                        COMP_names = DM_cross_names[COMP_dms]
                                        COMP_sigmas = DM_cross_sigmas[COMP_dms]
                                        COMP_powers = DM_cross_powers[COMP_dms]
                                        COMP_periods_ms = DM_cross_periods_ms[COMP_dms]
                                        COMP_individual_flags = DM_cross_individual_flags[COMP_dms]
                                        COMP_multi_flags = DM_cross_multi_flags[COMP_dms]
                                        COMP_N_cands = len(COMP_names)
                                        COMP_flag_nan = np.zeros(COMP_N_cands)

                                        COMP_DM = float(COMP_dms.replace("DM", "").replace("_","."))

                                        FLAG_DM = 0

                                        for j in range(COMP_N_cands):
                                                if (COMP_multi_flags[j] == -1):
                                                        Check_flag, Check_harm = check_cross(REF_periods_ms[i],COMP_periods_ms[j],fourier_bin,numharm,BIN_FACTOR)

                                                        if Check_flag:
                                                                FLAG_DM = 1
                                                                print("The candidate at DM %.2f with period %.5f ms has been removed because it is the %s harmonic of the candidate at DM %.2f with period %.5f ms" % (COMP_DM, COMP_periods_ms[j],Check_harm,REF_DM,REF_periods_ms[i]))
                                                                DM_cross_multi_flags[COMP_dms][j] = 0


                                                                Plot_names.append(COMP_names[j])
                                                                Plot_DMs.append(COMP_DM)
                                                                Plot_sigmas.append(COMP_sigmas[j])
                                                                Plot_powers.append(COMP_powers[j])
                                                                Plot_harms.append(Check_harm)

                                        Count_DM_cand = Count_DM_cand + FLAG_DM

                                if (len(Plot_names) == 0):
                                       print("ERROR IN THE CANDIDATE PLOT")
                                       exit()
                                if (len(Plot_names) == 1):
                                       if (REF_individual_flags[i] == 1):
                                                Single_name = REF_names[i]
                                       else:
                                                Single_name = "SINGLE_" + str(Count_single)
                                       Single_candidates_names.append(Single_name)

                                       Single_dms = Plot_DMs[0]
                                       Single_candidates_maxdms_sigma.append(Single_dms)
                                       Single_candidates_maxdms_power.append(Single_dms)

                                       Single_candidates_ndms.append(1)

                                       Single_candidates_sigmas.append(Plot_sigmas[0])

                                       Single_candidates_powers.append(Plot_powers[0])

                                       Single_candidates_periods_ms.append(REF_periods_ms[i])

                                       Count_single = Count_single + 1

                                if (len(Plot_names) > 1):

                                        fig = plt.figure(figsize=(10, 7))
                                        gs = gridspec.GridSpec(2, 1)
                                        ax_sigma = plt.subplot(gs[0, 0])
                                        ax_power = plt.subplot(gs[1, 0])

                                        if (REF_individual_flags[i] == 1):
                                                ax_sigma.set_title(REF_names[i]+"   P: " + str(REF_periods_ms[i])+" ms")
                                        else:
                                                ax_sigma.set_title(str(REF_periods_ms[i])+" ms")


                                        if (REF_individual_flags[i] == 1):
                                                Multi_name = REF_names[i]
                                        else:
                                                Multi_name = "MULTI_" + str(Count_Multi)
                                        Multi_candidates_names.append(Multi_name)

                                        Curr_Multi_ndms = Count_DM_cand
                                        Multi_candidates_ndms.append(Curr_Multi_ndms)
                                        print("Candidate %s detected in %d dms and the plot contains %d points." % (Multi_name,Curr_Multi_ndms,len(Plot_sigmas)))
                                        
                                        Multi_candidates_periods_ms.append(REF_periods_ms[i])

                                        Plot_harms_np = np.array(Plot_harms)
                                        Plot_DMs_np = np.array(Plot_DMs)
                                        Plot_sigmas_np = np.array(Plot_sigmas)
                                        Plot_powers_np = np.array(Plot_powers)

                                        unique_harms = np.unique(Plot_harms_np)


                                        ### Sigma plot
                                        
                                        Max_idx_sigma = np.argmax(Plot_sigmas)
                                        Curr_Multi_maxdm_sigma = Plot_DMs[Max_idx_sigma]
                                        Multi_candidates_maxdms_sigma.append(Curr_Multi_maxdm_sigma)

                                        Multi_maxsigma = np.max(Plot_sigmas)
                                        Multi_candidates_sigmas.append(Multi_maxsigma)

                                        markers_sigma = cycle(['o', 's', '^', 'v', 'D', 'p', 'h', 'H', '+', 'x', 'd', '|', '_', '8', '<', '>'])

                                        for curr_harms in unique_harms:
                                                if curr_harms == "1/1":
                                                       curr_mark = '*'
                                                else:
                                                       curr_mark = next(markers_sigma)
                                                ax_sigma.scatter(Plot_DMs_np[Plot_harms_np==curr_harms],Plot_sigmas_np[Plot_harms_np==curr_harms],marker=curr_mark,label=curr_harms)
                                        ax_sigma.legend()

                                        ### Power plot
                                        
                                        Max_idx_power = np.argmax(Plot_powers)
                                        Curr_Multi_maxdm_power = Plot_DMs[Max_idx_power]
                                        Multi_candidates_maxdms_power.append(Curr_Multi_maxdm_power)

                                        Multi_maxpower = np.max(Plot_powers)
                                        Multi_candidates_powers.append(Multi_maxpower)

                                        markers_power = cycle(['o', 's', '^', 'v', 'D', 'p', 'h', 'H', '+', 'x', 'd', '|', '_', '8', '<', '>'])

                                        for curr_harms in unique_harms:
                                                if curr_harms == "1/1":
                                                       curr_mark = '*'
                                                else:
                                                       curr_mark = next(markers_power)
                                                ax_power.scatter(Plot_DMs_np[Plot_harms_np==curr_harms],Plot_powers_np[Plot_harms_np==curr_harms],marker=curr_mark,label=curr_harms)
                                        ax_power.legend()

                                        ax_power.set_xlabel("DM [pc cm^-3]")
                                        ax_sigma.set_ylabel("Sigma")
                                        ax_power.set_ylabel("Power")
                                        plt.setp(ax_sigma.get_xticklabels(), visible=False)

                                        Plot_out = "RESULTS_DMCROSS/" + Multi_name
                                        plt.savefig(Plot_out, dpi=300)
                                        plt.close()

                                        Count_Multi = Count_Multi + 1


        Curr_Multi_results = pd.DataFrame({"Name":Multi_candidates_names,
                                   "Sigma_max":Multi_candidates_sigmas,
                                   "Power_max":Multi_candidates_powers,
                                   "NDMs":Multi_candidates_ndms,
                                   "MaxDMs_sigma":Multi_candidates_maxdms_sigma,
                                   "MaxDMs_power":Multi_candidates_maxdms_power,
                                   "Period_ms":Multi_candidates_periods_ms})
        
        Curr_single_results = pd.DataFrame({"Name":Single_candidates_names,
                                   "Sigma_max":Single_candidates_sigmas,
                                   "Power_max":Single_candidates_powers,
                                   "NDMs":Single_candidates_ndms,
                                   "MaxDMs_sigma":Single_candidates_maxdms_sigma,
                                   "MaxDMs_power":Single_candidates_maxdms_power,
                                   "Period_ms":Single_candidates_periods_ms})
        
        Curr_cross_results = pd.concat([Curr_Multi_results, Curr_single_results], ignore_index=True)
        Name_cross_out = "RESULTS_DMCROSS/STACK_CAND_DMCROSS.csv"
        Curr_cross_results.to_csv(Name_cross_out,index=False)
        if VERBOSITY:
                pd.set_option('display.width', 1000)
                pd.set_option('display.max_columns', None)  # Show all columns
                pd.set_option('display.max_rows', None)     # Show all rows (optional)

                print(Curr_cross_results)


def check_cross(REF_period_ms,COMP_period_ms,fourier_bin,numharm,BIN_FACTOR):

        REF_period_s = REF_period_ms / 1000

        REF_freq = 1 / REF_period_s
        REF_freq_low = REF_freq - (BIN_FACTOR * fourier_bin)
        REF_freq_high = REF_freq + (BIN_FACTOR * fourier_bin)

        COMP_period_s = COMP_period_ms / 1000
        COMP_freq_hz = 1 / COMP_period_s
       
        if (COMP_freq_hz > REF_freq_low) and (COMP_freq_hz < REF_freq_high):
                Curr_harm_str = "1/1"
                return True , Curr_harm_str

        else:
                # ----------------- HARMONICS ---------------
                for nh in range(1, numharm + 1):
                        for n in range(1, numharm+1):
                                REF_freq_nh_hz = REF_freq * (np.float64(nh) / n) # opposite fraction because frequency
                                REF_freq_nh_low = REF_freq_nh_hz - (BIN_FACTOR * fourier_bin)
                                REF_freq_nh_high = REF_freq_nh_hz + (BIN_FACTOR * fourier_bin)


                                if (COMP_freq_hz >= REF_freq_nh_low) and (COMP_freq_hz <= REF_freq_nh_high):
                                        Curr_harm_str = "%d/%d" % (n,nh)
                                        return True , Curr_harm_str

        return False , "ERROR"


def remove_harmonics(REF_period_ms, periods_list_ms, REF_flag, Flag_cands, Name_cands, numharm, Curr_num, Curr_den, fourier_bin, BIN_FACTOR, known_freq_low, known_freq_high,REF_idx):
        
        freq_list_hz = 1000 / periods_list_ms
        
        # NEED TO PUT ZEROS ON ALL THE HARMONICS AND IF THE PULSAR IS KNOWN BUT NOT IN THE FUNDAMENTAL HARMONIC IT HAS TO MOVE THE 1 IF IT FINDS A BETTER CANDIDATE
        REF_period_s = REF_period_ms / 1000
        REF_freq = 1 / REF_period_s

        if REF_flag == 1: # Known pulsar
                REF_freq_low_hz = known_freq_low
                REF_freq_high_hz = known_freq_high
        
        if REF_flag == 2: # Interesting candidate not known
                REF_freq_low_hz = REF_freq - (BIN_FACTOR * fourier_bin)
                REF_freq_high_hz = REF_freq + (BIN_FACTOR * fourier_bin)


        for cand_idx in range(len(periods_list_ms)):
                if Flag_cands[cand_idx] == -1:

                        if (freq_list_hz[cand_idx] > REF_freq_low_hz) and (freq_list_hz[cand_idx] < REF_freq_high_hz):
                               Flag_cands[cand_idx] = 0
                               print("Candidate %d with period %.4f ms removed because in the same interval of candidate %d with period %.4f ms" % (cand_idx,periods_list_ms[cand_idx],REF_idx,REF_period_ms))
                        
                        else:
                                # ----------------- HARMONICS ---------------
                                for nh in range(1, numharm + 1):
                                        for n in range(1, numharm+1):
                                                REF_freq_nh = REF_freq * (np.float64(nh) / n)
                                                REF_freq_nh_low = REF_freq_nh - (BIN_FACTOR * fourier_bin)
                                                REF_freq_nh_high = REF_freq_nh + (BIN_FACTOR * fourier_bin)

                                                if (freq_list_hz[cand_idx] >= REF_freq_nh_low) and (freq_list_hz[cand_idx] <= REF_freq_nh_high):
                                                        if (nh == Curr_num) and (n == Curr_den):
                                                              # SWITCHING TO THE FUNDAMENTAL HARMONIC OF THE KNOW PULSAR
                                                              Flag_cands[cand_idx] = 1
                                                              Flag_cands[REF_idx] = 0
                                                              
                                                              Name_cands[cand_idx] = Name_cands[REF_idx]
                                                              Name_cands[REF_idx] = "0.0"
                                                              break
                                                        else:
                                                              Flag_cands[cand_idx] = 0
                                                              print("Candidate %d with period %.4f ms removed because in the interval of the %d/%d harmonic of candidate %d with period %.4f ms" % (cand_idx,periods_list_ms[cand_idx],n,nh,REF_idx,REF_period_ms))
                                                              break

        return Flag_cands, Name_cands


def import_known_pulsars(PAR_FOLD, Tstart_MJD, Fourier_bin, BIN_FACTOR):
        ################################################################################
        #   IMPORT PARFILES OF KNOWN PULSARS
        ################################################################################

        # PAR_FOLD is the folder containing the .par files
        # Tstart_MJD is a median mjd in the sample of observations
        # Fourier_bin is a median size (in Hz) of the fourier bins in the observations

        list_known_pulsars = []
        freq_low_lim = []
        freq_high_lim = []
        if os.path.exists(PAR_FOLD):
                list_parfilenames = sorted(glob.glob("%s/*.par" % PAR_FOLD))

                for k in range(len(list_parfilenames)):
                        current_pulsar = Pulsar(list_parfilenames[k])
                        list_known_pulsars.append(current_pulsar)

                        current_freq = psr_utils.calc_freq(Tstart_MJD, current_pulsar.PEPOCH, current_pulsar.F0, current_pulsar.F1, current_pulsar.F2) # COMPUTE FREQUENCY AT TIME OF OBSERVATION
                        current_p = 1 / current_freq # Period
                        current_p_short = current_p * (1 - current_pulsar.doppler_factor) # SHORTER PERIOD DUE TO DOPPLER
                        current_p_long = current_p * (1 + current_pulsar.doppler_factor) # LONGER PERIOD DUE TO DOPPLER
                        current_freq_low_lim = (1/current_p_long) - (Fourier_bin * BIN_FACTOR) # LOWER FREQUENCY DUE TO DOPPLER AND ADDING FOURIER BIN UNCERTAINTY
                        current_freq_high_lim = (1/current_p_short) + (Fourier_bin * BIN_FACTOR) # HIGHER FREQUENCY DUE TO DOPPLER AND ADDING FOURIER BIN UNCERTAINTY

                        freq_low_lim.append(current_freq_low_lim)
                        freq_high_lim.append(current_freq_high_lim)

 
        else:
               print("FOLDER WITH KNOWN PULSARS NOT FOUND")

        list_known_pulsars = np.array(list_known_pulsars)
        freq_low_lim = np.array(freq_low_lim)
        freq_high_lim = np.array(freq_high_lim)

        sorted_indices = np.argsort(freq_low_lim) # sort ascending

        list_known_pulsars = list_known_pulsars[sorted_indices]
        freq_low_lim = freq_low_lim[sorted_indices]
        freq_high_lim = freq_high_lim[sorted_indices]

        return list_known_pulsars, freq_low_lim, freq_high_lim


def check_if_cand_is_known(P_cand_ms, list_known_pulsars,known_freq_low,known_freq_high, index, numharm):
        # Loop over all the known periods of the pulsars in the cluster
        print("---------------------------------------------------------------")
        print("---------------------------------------------------------------")
        print("Period (ms): ",P_cand_ms)

        for i in range(len(list_known_pulsars)):
                P_ms = list_known_pulsars[i].P0_ms
                psrname = list_known_pulsars[i].psr_name
                P_ms_min = (1/known_freq_high[i]) * 1000
                P_ms_max = (1/known_freq_low[i]) * 1000


                if (P_cand_ms > P_ms_min) and (P_cand_ms < P_ms_max):
                        curr_num = 1
                        curr_den = 1
                        if VERBOSITY:
                               print("Candidate %d is the funtamental harmonic of pulsar %s at period %.7f ms" % (index,psrname,P_ms))
                        return True, psrname, curr_num, curr_den , int(i)

                else:
                        # ----------------- HARMONICS ---------------
                        for nh in range(1, numharm + 1):
                                for n in range(1, numharm+1):
                                        P_known_ms_nh_min = P_ms_min * (np.float64(n) / nh)
                                        P_known_ms_nh_max = P_ms_max * (np.float64(n) / nh)

                                        if (P_cand_ms >= P_known_ms_nh_min) and (P_cand_ms <= P_known_ms_nh_max):
                                                if VERBOSITY:
                                                        print("Candidate %d with period %.7f is the %d/%d-th harmonic of pulsar %s with period %.7f ms" % (index,P_cand_ms,n,nh,psrname,P_ms))
                                                return True, psrname, n, nh , int(i)

        print("No match")
        return False, "None", -1, -1, -1

def prepfold_command_from_csv(Results_dir,root_dir,Longest_file,n_cpu=None):
        os.chdir(Results_dir)
        Curr_result_file = pd.read_csv("STACK_CAND_DMCROSS.csv")
        Curr_names = np.array(Curr_result_file.Name).astype("str")
        Curr_sigmas = np.array(Curr_result_file.Sigma_max).astype("float")
        Curr_periods_ms = np.array(Curr_result_file.Period_ms).astype("float")
        Curr_maxdms = np.array(Curr_result_file.MaxDMs_sigma).astype("float")

        for i in range(len(Curr_result_file)):
                Cand_name = Curr_names[i]
                if not os.path.exists(Cand_name):
                        os.mkdir(Cand_name)

                Curr_dir = Results_dir+Cand_name+"/"
                
                os.chdir(Curr_dir)
                print("Ora sono in %s" % (os.getcwd()))

                Cand_DM = Curr_maxdms[i]
                Cand_p_ms = Curr_periods_ms[i]     

                file_basename = os.path.basename(Longest_file)
                file_basename = os.path.splitext(file_basename)[0]

                # Search for maskfile in root_dir
                maskfile_list = glob.glob(os.path.join(root_dir, f"{file_basename}*rfifind*.mask"))

                Curr_name = file_basename + "_" + str(Cand_DM).replace(".","_")

                if len(maskfile_list) > 0:
                        maskfile = maskfile_list[0]
                        cmd_prep = f"prepfold -p {Cand_p_ms/1000.0} -dm {Cand_DM} {Longest_file} " \
                                f"-noscales -nooffsets -mask {maskfile} -ncpus {str(n_cpu)} -n 128 -noxwin -o {Curr_name}"
                else:
                        # No maskfile found - run without mask
                        print(f"Warning: No maskfile found for {file_basename}")
                        cmd_prep = f"prepfold -p {Cand_p_ms/1000.0} -dm {Cand_DM} {Longest_file} " \
                                f"-noscales -nooffsets -ncpus {str(n_cpu)} -n 128 -noxwin -o {Curr_name}"

                print("Doing prepfold:")
                print(cmd_prep)
                os.system(cmd_prep)



                os.chdir(Results_dir)
                print("Ora sono in %s" % (os.getcwd()))



if __name__ == "__main__":
    

    parser = OptionParser(description='Script that coherently stacks multiple observations.')

    parser.set_usage('Coherent_stack.py [options]')

    parser.add_option("--FILES", action="store", dest="FILES",
                      default=None, help="List of files to be analysed (comma separated).")
    
    parser.add_option("--FILES_FOLDER", action="store", dest="FILES_FOLDER",
                      default=None, help="Folder that contains the files to be analysed in .fits or .fil (alternative to the --FILES option).")
    
    parser.add_option("--DM_RANGE", action="store", dest="DM_RANGE",
                      default=None, help="Minimum and maximum dispersion measures [pc cm^-3] in the search (comma separated).")
    
    parser.add_option("--DM_COHERENT", action="store", dest="DM_COHERENT",
                      default=0, help="Dispersion measures [pc cm^-3] of coherent de-dispersion.")
    
    parser.add_option("--STACK_THRESH", action="store", dest="STACK_THRESH",
                      default=7, help="Sigma cut-off of candidates in incoherent stacksearch (Default: 7).")
    
    parser.add_option("--STACK_MAXCANDS", action="store", dest="STACK_MAXCANDS",
                      default=1000, help="Maximum number of candidates in incoherent stacksearch (Default: 1000).")
    
    parser.add_option("--STACK_NHARMS", action="store", dest="STACK_NHARMS",
                      default=32, help="Maximum number of harmonics to sum in incoherent stacksearch (Default: 32).")
    
    parser.add_option("--KNOWN_PULSARS_FOLDER", action="store", dest="KNOWN_PULSARS_FOLDER",
                      default=None, help="A folder containing the .par files of known pulsars.")
    
    parser.add_option("--NCPUS", action="store", dest="NCPUS",
                      default=4, help="Number of CPUs used for the computation (Default: 4).")
    
    parser.add_option("--BIN_TOLERANCE", action="store", dest="BIN_TOLERANCE",
                      default=1.1, help="Tolerance in candidates harmonic identification. Default: 1.1, that means checking on an interval of +- 1.1 * fourier bin in Hz.")
    
    parser.add_option("--v", action="store_true", dest="VERBOSITY",
                      default=False, help="Enable verbosity.")
    
    parser.add_option("--enable_stack_different", action="store_true", dest="STACK_DIFF",
                      default=False, help="Enable stacking of observations with different parmeters (Central freq, nchan, bandwidth and time sampling).")
    
    parser.add_option("--NO_PREPSUBBAND", action="store_false", dest="PREPSUB_FLAG",
                      default=True, help="Skip prepsubband.")
    

    (conf, args) = parser.parse_args(argv[1:])

    # Input parameters
    ###################################################################

    VERBOSITY = conf.VERBOSITY
    STACK_DIFF = conf.STACK_DIFF
    DM_COHERENT = float(conf.DM_COHERENT)
    NCPUS = int(conf.NCPUS)
    STACK_THRESH = float(conf.STACK_THRESH)
    STACK_MAXCANDS = int(conf.STACK_MAXCANDS)
    STACK_NHARMS = int(conf.STACK_NHARMS)
    BIN_TOLERANCE = float(conf.BIN_TOLERANCE)
    PREPSUB_FLAG = conf.PREPSUB_FLAG

    KNOWN_PULSARS_FOLDER = str(conf.KNOWN_PULSARS_FOLDER)


    ###################################################################

    FILES = str(conf.FILES)
    FILES_FOLDER = str(conf.FILES_FOLDER)

    if FILES != "None" and FILES_FOLDER != "None":
        print("Specify only a list of files OR a folder containing the files, not both.")
    elif FILES != "None":
        try:
            FILES_LIST = FILES.split(",")
            if len(FILES_LIST) == 1:
                print("Provide at least 2 files.")
                exit()

        except:
            print("Provide a proper list of files.")
            exit()

    elif FILES_FOLDER != "None":
        FILES_LIST = []
        for root,dirs,files in os.walk(FILES_FOLDER):
            for Curr_file in files:
                if Curr_file.endswith(".fits") or Curr_file.endswith(".fil"):
                    FILES_LIST.append(Curr_file)

        if len(FILES_LIST) == 1:
                print("The folder needs to contain at least 2 files.")
                exit()

    N_OBS = len(FILES_LIST)
    list_observations = []
    for OBS_idx in range(N_OBS):
        if FILES_LIST[OBS_idx].endswith(".fits"):
            Curr_datatype = "psrfits"
        elif FILES_LIST[OBS_idx].endswith(".fil"):
            Curr_datatype = "filterbank"

        list_observations.append(Observation(FILES_LIST[OBS_idx], Curr_datatype))

    ###################################################################
    DM_RANGE = str(conf.DM_RANGE)

    try:
        DM_RANGE = np.array(DM_RANGE.split(",")).astype("float")
        if len(DM_RANGE) != 2:
            print("Provide a proper range of DMs.")
            exit()
            # Tstart_TOT = np.array((Tstart_TOT)).reshape(1,)

    except:
        print("Provide a proper range of DMs.")
        exit()

    DM_MIN = np.min(DM_RANGE)
    DM_MAX = np.max(DM_RANGE)
    ###################################################################

    

    ###################################################################



    reference_freq_central_MHz    = list_observations[0].freq_central_MHz
    reference_nchan               = list_observations[0].nchan
    reference_bw_MHz              = list_observations[0].bw_MHz
    reference_t_samp_s            = list_observations[0].t_samp_s
    maximum_length_s              = 0
    j_longest_observation         = -1
    minimum_length_s              = 10e15
    j_shortest_observation         = -1

    obs_mjds = []
    for i in range(len(list_observations)):
           obs_mjds.append(list_observations[i].Tstart_MJD)

    reference_mjd = np.median(obs_mjds)

    root_dir = os.getcwd()

    if VERBOSITY:
        print()
        print("     %-40s %20s %15s %10s %12s %12s" % ("Observations to search:", "Length (s)", "Freq (MHz)", "Nchan", "BW (MHz)", "tsamp (s)"))
    for j in range(N_OBS):
        obs = list_observations[j]
        print("%3d) %-40s %20.3f %15.3f %10d %12.3f %12.3e" % (j+1, FILES_LIST[j], obs.T_obs_s, obs.freq_central_MHz, obs.nchan, obs.bw_MHz, obs.t_samp_s))

        if (obs.freq_central_MHz != reference_freq_central_MHz):
                if STACK_DIFF:
                       print("WARNING: %s has a different central frequency (%.2f MHz != %.2f MHz)" % (FILES_LIST[j], obs.freq_central_MHz, reference_freq_central_MHz))
                else:
                        print("ERROR: %s has a different central frequency (%.2f MHz != %.2f MHz)" % (FILES_LIST[j], obs.freq_central_MHz, reference_freq_central_MHz))
                        exit()
        if (obs.nchan != reference_nchan):
                if STACK_DIFF:
                       print("WARNING: %s has a different number of channels (%d != %d)" % (FILES_LIST[j], obs.nchan, reference_nchan))
                else:
                        print("ERROR: %s has a different number of channels (%d != %d)" % (FILES_LIST[j], obs.nchan, reference_nchan))
                        exit()                        
        if (obs.bw_MHz != reference_bw_MHz):
                if STACK_DIFF:
                       print("WARNING: %s has a different bandwidth (%.2f MHz != %.2f MHz)" % (FILES_LIST[j], obs.bw_MHz, reference_bw_MHz))
                else:
                        print("ERROR: %s has a different bandwidth (%.2f MHz != %.2f MHz)" % (FILES_LIST[j], obs.bw_MHz, reference_bw_MHz))
                        exit()

        if (obs.t_samp_s != reference_t_samp_s):
                if STACK_DIFF:
                       print("WARNING: %s has a different sampling time (%.3e s != %.3e s)" % (FILES_LIST[j], obs.t_samp_s, reference_t_samp_s))
                else:
                        print("ERROR: %s has a different sampling time (%.3e s != %.3e s)" % (FILES_LIST[j], obs.t_samp_s, reference_t_samp_s))
                        exit()

        if obs.T_obs_s > maximum_length_s:
                maximum_length_s = obs.T_obs_s
                j_longest_observation = j
        
        if obs.T_obs_s < minimum_length_s:
                minimum_length_s = obs.T_obs_s
                j_shortest_observation = j

    print()
    print("GOOD! All files seem to have the same observing setup!")
    print()
    print("Longest observation: %d) '%s' (%.3f s)" % (j_longest_observation+1, FILES_LIST[j_longest_observation], maximum_length_s))

    longest_file = root_dir + "/" + FILES_LIST[j_longest_observation]

    N_samples_max_length = int(maximum_length_s / reference_t_samp_s)
    N_samples_numout = int( (N_samples_max_length / 10000) + 1) *  10000
    print("N_samples_numout = %d = %.3f s" % (N_samples_numout, N_samples_numout*reference_t_samp_s))


    reference_fourier_bin = 1 / minimum_length_s # Fourier bin of the shortest observation
    list_known_pulsars, list_known_freq_low_lim, list_known_freq_high_lim = import_known_pulsars(KNOWN_PULSARS_FOLDER, reference_mjd, reference_fourier_bin, BIN_TOLERANCE)

    if VERBOSITY and len(list_known_freq_low_lim) > 0:
        print("PULSAR NAMES AND FREQUENCIES AND PERIODS")
        for i in range(len(list_known_freq_low_lim)):
                print(list_known_pulsars[i].psr_name)
        print(list_known_freq_low_lim)
        print(list_known_freq_high_lim)
        print(1/list_known_freq_low_lim)
        print(1/list_known_freq_high_lim)

    log_filename = "LOG-alex_stack_search_pipeline_%s.txt" % (datetime.datetime.now()).strftime("%Y%m%d_%H%M")

    list_DDplan_scheme = get_DDplan_scheme(list_observations[0].file_abspath,
                                    root_dir,
                                    log_filename,
                                    DM_MIN,
                                    DM_MAX,
                                    DM_COHERENT,
                                    reference_freq_central_MHz,
                                    reference_bw_MHz,
                                    reference_nchan,
                                    0,
                                    reference_t_samp_s)


    #################################################
    # 1)  DEDISPERSE WITH PREPSUBBAND
    #################################################

    TP = ThreadPool(NCPUS)
    for j in range(N_OBS):
        obs = list_observations[j]
        obs_log_filename = log_filename.replace(".log", "_%s.log" % obs.file_basename)
        try:
                maskfile = glob.glob("%s*rfifind*.mask" % (obs.file_basename))[0]
                print("maskfile = ", maskfile)
        except:
                maskfile = ""
                print("maskfile not present for ", obs.file_basename)

        if PREPSUB_FLAG:
                TP.apply_async(
                prepsubband,
                args=(
                        obs.file_abspath,
                        root_dir,
                        obs_log_filename,
                        "full",
                        "ck00",
                        N_samples_numout,
                        maskfile,
                        list_DDplan_scheme,
                        reference_nchan,
                        0,
                        "",
                ),
                kwds={
                        "verbosity_level": int(VERBOSITY),
                        "n_cpu": 1
                }, error_callback=handle_error
                )

    TP.close()
    TP.join()


    print(list_DDplan_scheme)
    print()
    N_schemes = len(list_DDplan_scheme)
    list_dm_dir = []
    for i in range(N_schemes):
            for dm in np.arange(list_DDplan_scheme[i]['loDM'],   list_DDplan_scheme[i]['highDM'] - 0.5*list_DDplan_scheme[i]['dDM']        , list_DDplan_scheme[i]['dDM']):
                    dir_dm = "DM%.2f" % (dm)
                    dir_dm = dir_dm.replace(".", "_")
                    list_dm_dir.append(dir_dm)

    print()
    print()

    for i in range(len(list_dm_dir)):
        dir_dm = list_dm_dir[i]
        dm = float(dir_dm.replace("DM", "").replace("_","."))
        if not os.path.exists(dir_dm):
                os.mkdir(dir_dm)
        print("dm = %.2f" % dm) 
        for f in glob.glob("*DM%s*.dat" % ("%.2f" % dm).replace(".","_")) + glob.glob("*DM%s*.inf" % ("%.2f" % dm).replace(".","_")):
                print("DM = %.2f --- Moving %s to %s/%s/%s" % (dm, f, root_dir, dir_dm, f ))
                os.rename("./%s" % f , "%s/%s/%s" % (root_dir, dir_dm, f))

    #################################################
    # 2)  MAKE FFTs
    #################################################
    for i in range(len(list_dm_dir)):
        
        dir_dm = list_dm_dir[i]
        dm = float(dir_dm.replace("DM", "").replace("_","."))
        workdir_current_dm = "%s/%s" % (root_dir, dir_dm)
        os.chdir(workdir_current_dm)
        print("Ora sono in %s" % (os.getcwd()))



        # CHECK IF STACK RESULTS ARE PRESENT
        Name_stack = dir_dm + "_STACK_CAND_INDIVIDUAL.csv"
        if os.path.exists(Name_stack) and (os.path.getsize(Name_stack) > 0):
                if VERBOSITY:
                        print()
                        print("WARNING: File %s already present. Skipping realfft, rednoise and stacksearch..." % (Name_stack))
        else:
        
                datfiles = glob.glob("*DM%s*.dat" % ("%.2f" % dm).replace(".", "_"))

                # Parallelize realfft+rednoise across .dat files
                TP2 = ThreadPool(NCPUS)
                for datfile in datfiles:
                        TP2.apply_async(
                        realfft_and_rednoise,
                        args=(datfile, workdir_current_dm, log_filename, VERBOSITY),
                        error_callback=handle_error
                        )
                TP2.close()
                TP2.join()  # wait for ALL realfft+rednoise to finish before stacksearch

                print("Doing stacksearch on %s..." % (os.getcwd()))
                Curr_fft_list = []


                for fftfile in glob.glob("*DM%s*_red.fft" % ("%.2f" % dm).replace(".","_")):
                        Curr_fft_list.append(fftfile)

                stacksearch(Curr_fft_list,
                                workdir_current_dm,
                                log_filename,
                                verbosity_level=int(VERBOSITY),
                                threshold = STACK_THRESH,
                                maxcands = STACK_MAXCANDS,
                                nharms = STACK_NHARMS
                        )
                
                for file in glob.glob("*.fft"):
                        os.remove(file)

                for file in glob.glob("*.dat"):
                        os.remove(file)
                
        stacksearch_results_individual(
                        verbosity_level=int(VERBOSITY),
                        nharms = STACK_NHARMS,
                        known_pulsars = list_known_pulsars,
                        known_freq_low = list_known_freq_low_lim,
                        known_freq_high = list_known_freq_high_lim,
                        dir_dm = dir_dm,
                        REF_fourier_bin = reference_fourier_bin,
                        STACK_TH = STACK_THRESH,
                        BIN_FACTOR = BIN_TOLERANCE
                        )

    print()
    #################################################
    # 3)  INCOHERENT FFTs STACKING
    #################################################
        
    os.chdir(root_dir)
        
    stacksearch_results_DMcross(DM_list=list_dm_dir, verbosity_level=int(VERBOSITY), fourier_bin = reference_fourier_bin, numharm=STACK_NHARMS, Min_threshold=STACK_THRESH, BIN_FACTOR = BIN_TOLERANCE)

    Results_dir = root_dir + "/RESULTS_DMCROSS/"

    #prepfold_command_from_csv(Results_dir,root_dir+"/",longest_file)

    print("done!"); sys.stdout.flush()


    exit()

