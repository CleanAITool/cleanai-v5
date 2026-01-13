# Windows Donanim Bilgileri Raporu
# Detayli sistem donanim bilgilerini toplar ve gosterir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   WINDOWS DONANIM BILGILERI RAPORU" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Isletim Sistemi Bilgileri
Write-Host "=== ISLETIM SISTEMI ===" -ForegroundColor Yellow
$os = Get-CimInstance Win32_OperatingSystem
Write-Host "Isletim Sistemi: $($os.Caption)"
Write-Host "Versiyon: $($os.Version)"
Write-Host "Mimari: $($os.OSArchitecture)"
Write-Host "Kurulum Tarihi: $($os.InstallDate)"
Write-Host "Son Acilis: $($os.LastBootUpTime)"
Write-Host "Sistem Dizini: $($os.SystemDirectory)"
Write-Host ""

# Bilgisayar Bilgileri
Write-Host "=== BILGISAYAR BILGILERI ===" -ForegroundColor Yellow
$cs = Get-CimInstance Win32_ComputerSystem
Write-Host "Bilgisayar Adi: $($cs.Name)"
Write-Host "Uretici: $($cs.Manufacturer)"
Write-Host "Model: $($cs.Model)"
Write-Host "Sistem Tipi: $($cs.SystemType)"
Write-Host "Domain: $($cs.Domain)"
Write-Host ""

# Islemci Bilgileri
Write-Host "=== ISLEMCI (CPU) ===" -ForegroundColor Yellow
$cpu = Get-CimInstance Win32_Processor
foreach ($proc in $cpu) {
    Write-Host "Islemci Adi: $($proc.Name)"
    Write-Host "Uretici: $($proc.Manufacturer)"
    Write-Host "Cekirdek Sayisi: $($proc.NumberOfCores)"
    Write-Host "Mantiksal Islemci: $($proc.NumberOfLogicalProcessors)"
    Write-Host "Maksimum Hiz: $($proc.MaxClockSpeed) MHz"
    Write-Host "Mevcut Hiz: $($proc.CurrentClockSpeed) MHz"
    Write-Host "L2 Onbellek: $($proc.L2CacheSize) KB"
    Write-Host "L3 Onbellek: $($proc.L3CacheSize) KB"
    Write-Host "Soket Tanimi: $($proc.SocketDesignation)"
    Write-Host ""
}

# Bellek (RAM) Bilgileri
Write-Host "=== BELLEK (RAM) ===" -ForegroundColor Yellow
Write-Host "Toplam Fiziksel Bellek: $([math]::Round($cs.TotalPhysicalMemory / 1GB, 2)) GB"
$memory = Get-CimInstance Win32_PhysicalMemory
$totalSlots = 0
foreach ($mem in $memory) {
    $totalSlots++
    Write-Host ""
    Write-Host "Bellek Slot ${totalSlots}:"
    Write-Host "  Kapasite: $([math]::Round($mem.Capacity / 1GB, 2)) GB"
    Write-Host "  Hiz: $($mem.Speed) MHz"
    Write-Host "  Uretici: $($mem.Manufacturer)"
    Write-Host "  Part Number: $($mem.PartNumber)"
    Write-Host "  Seri No: $($mem.SerialNumber)"
    Write-Host "  Form Faktor: $($mem.FormFactor)"
    Write-Host "  Bellek Tipi: $($mem.MemoryType)"
}
Write-Host ""

# Anakart Bilgileri
Write-Host "=== ANAKART (MOTHERBOARD) ===" -ForegroundColor Yellow
$mb = Get-CimInstance Win32_BaseBoard
Write-Host "Uretici: $($mb.Manufacturer)"
Write-Host "Urun: $($mb.Product)"
Write-Host "Seri No: $($mb.SerialNumber)"
Write-Host "Versiyon: $($mb.Version)"
Write-Host ""

# BIOS Bilgileri
Write-Host "=== BIOS ===" -ForegroundColor Yellow
$bios = Get-CimInstance Win32_BIOS
Write-Host "Uretici: $($bios.Manufacturer)"
Write-Host "Versiyon: $($bios.Version)"
Write-Host "Seri No: $($bios.SerialNumber)"
Write-Host "Tarih: $($bios.ReleaseDate)"
Write-Host ""

# Disk Bilgileri
Write-Host "=== DISKLER (STORAGE) ===" -ForegroundColor Yellow
$disks = Get-CimInstance Win32_DiskDrive
foreach ($disk in $disks) {
    Write-Host ""
    Write-Host "Disk: $($disk.Caption)"
    Write-Host "  Model: $($disk.Model)"
    Write-Host "  Boyut: $([math]::Round($disk.Size / 1GB, 2)) GB"
    Write-Host "  Arayuz: $($disk.InterfaceType)"
    Write-Host "  Bolum Sayisi: $($disk.Partitions)"
    Write-Host "  Seri No: $($disk.SerialNumber)"
    Write-Host "  Durum: $($disk.Status)"
}
Write-Host ""

# Bolum (Partition) Bilgileri
Write-Host "=== BOLUMLER (PARTITIONS) ===" -ForegroundColor Yellow
$volumes = Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3"
foreach ($vol in $volumes) {
    Write-Host ""
    Write-Host "Surucu: $($vol.DeviceID)"
    Write-Host "  Dosya Sistemi: $($vol.FileSystem)"
    Write-Host "  Toplam Boyut: $([math]::Round($vol.Size / 1GB, 2)) GB"
    Write-Host "  Bos Alan: $([math]::Round($vol.FreeSpace / 1GB, 2)) GB"
    Write-Host "  Kullanilan: $([math]::Round(($vol.Size - $vol.FreeSpace) / 1GB, 2)) GB"
    Write-Host "  Kullanim Orani: $([math]::Round((($vol.Size - $vol.FreeSpace) / $vol.Size) * 100, 2))%"
    Write-Host "  Etiket: $($vol.VolumeName)"
}
Write-Host ""

# Ekran Karti (GPU) Bilgileri
Write-Host "=== EKRAN KARTI (GPU) ===" -ForegroundColor Yellow

# NVIDIA GPU icin nvidia-smi ile bilgi al
$nvidiaGPUs = @()
try {
    $nvidiaSmi = nvidia-smi --query-gpu=name,memory.total,driver_version,temperature.gpu --format=csv,noheader,nounits 2>$null
    if ($nvidiaSmi) {
        foreach ($line in $nvidiaSmi) {
            $parts = $line -split ','
            if ($parts.Count -ge 4) {
                $nvidiaGPUs += @{
                    Name = $parts[0].Trim()
                    Memory = $parts[1].Trim()
                    Driver = $parts[2].Trim()
                    Temp = $parts[3].Trim()
                }
            }
        }
    }
} catch {
    # nvidia-smi bulunamazsa devam et
}

# Paylasilan GPU bellegi icin sistem RAM bilgisi
$sharedMemoryGB = [math]::Round($cs.TotalPhysicalMemory / 2 / 1GB, 1)

$gpu = Get-CimInstance Win32_VideoController
$gpuIndex = 0
foreach ($card in $gpu) {
    Write-Host ""
    Write-Host "Ekran Karti: $($card.Name)"
    Write-Host "  Uretici: $($card.AdapterCompatibility)"
    
    # NVIDIA kart ise nvidia-smi bilgisini kullan
    $isNvidia = $card.Name -like "*NVIDIA*"
    if ($isNvidia -and $nvidiaGPUs.Count -gt 0) {
        $matchingNvidiaGPU = $nvidiaGPUs | Where-Object { $card.Name -like "*$($_.Name)*" } | Select-Object -First 1
        if ($matchingNvidiaGPU) {
            $dedicatedGB = [math]::Round([double]$matchingNvidiaGPU.Memory / 1024, 2)
            $totalGB = [math]::Round($dedicatedGB + $sharedMemoryGB, 1)
            Write-Host "  Adanmis GPU Bellegi: $($matchingNvidiaGPU.Memory) MB ($dedicatedGB GB)"
            Write-Host "  Paylasilan GPU Bellegi: $sharedMemoryGB GB (Sistem RAM)"
            Write-Host "  Toplam GPU Bellegi: $totalGB GB"
            Write-Host "  Sicaklik: $($matchingNvidiaGPU.Temp) C"
        } else {
            Write-Host "  Video Bellegi: $([math]::Round($card.AdapterRAM / 1GB, 2)) GB"
        }
    } else {
        $memGB = [math]::Round($card.AdapterRAM / 1GB, 2)
        if ($memGB -gt 0) {
            Write-Host "  Adanmis GPU Bellegi: $memGB GB"
            Write-Host "  Paylasilan GPU Bellegi: $sharedMemoryGB GB (Sistem RAM)"
            Write-Host "  Toplam GPU Bellegi: $([math]::Round($memGB + $sharedMemoryGB, 1)) GB"
        } else {
            $memMB = [math]::Round($card.AdapterRAM / 1MB, 2)
            Write-Host "  Adanmis GPU Bellegi: $memMB MB"
            Write-Host "  Paylasilan GPU Bellegi: $sharedMemoryGB GB (Sistem RAM)"
        }
    }
    
    Write-Host "  Surucu Versiyonu: $($card.DriverVersion)"
    if ($card.CurrentHorizontalResolution -and $card.CurrentVerticalResolution) {
        Write-Host "  Mevcut Cozunurluk: $($card.CurrentHorizontalResolution) x $($card.CurrentVerticalResolution)"
        Write-Host "  Yenileme Hizi: $($card.CurrentRefreshRate) Hz"
    }
    Write-Host "  Durum: $($card.Status)"
    $gpuIndex++
}
Write-Host ""

# Monitor Bilgileri
Write-Host "=== MONITOR ===" -ForegroundColor Yellow
$monitors = Get-CimInstance WmiMonitorID -Namespace root\wmi -ErrorAction SilentlyContinue
if ($monitors) {
    foreach ($mon in $monitors) {
        $manufacturer = [System.Text.Encoding]::ASCII.GetString($mon.ManufacturerName -ne 0)
        $name = [System.Text.Encoding]::ASCII.GetString($mon.UserFriendlyName -ne 0)
        $serial = [System.Text.Encoding]::ASCII.GetString($mon.SerialNumberID -ne 0)
        Write-Host "Uretici: $manufacturer"
        Write-Host "Model: $name"
        Write-Host "Seri No: $serial"
        Write-Host ""
    }
} else {
    Write-Host "Monitor bilgisi alinamadi" -ForegroundColor Red
    Write-Host ""
}

# Ag Adaptorleri
Write-Host "=== AG ADAPTORLERI ===" -ForegroundColor Yellow
$adapters = Get-CimInstance Win32_NetworkAdapter | Where-Object {$_.PhysicalAdapter -eq $true}
foreach ($adapter in $adapters) {
    Write-Host ""
    Write-Host "Adaptor: $($adapter.Name)"
    Write-Host "  MAC Adresi: $($adapter.MACAddress)"
    Write-Host "  Hiz: $($adapter.Speed)"
    Write-Host "  Uretici: $($adapter.Manufacturer)"
    Write-Host "  Durum: $($adapter.NetConnectionStatus)"
    
    $config = Get-CimInstance Win32_NetworkAdapterConfiguration | Where-Object {$_.Index -eq $adapter.Index}
    if ($config.IPAddress) {
        Write-Host "  IP Adresi: $($config.IPAddress -join ', ')"
        Write-Host "  Subnet Mask: $($config.IPSubnet -join ', ')"
        Write-Host "  Gateway: $($config.DefaultIPGateway -join ', ')"
        Write-Host "  DNS Sunucular: $($config.DNSServerSearchOrder -join ', ')"
        Write-Host "  DHCP Aktif: $($config.DHCPEnabled)"
    }
}
Write-Host ""

# Ses Karti
Write-Host "=== SES KARTI ===" -ForegroundColor Yellow
$sound = Get-CimInstance Win32_SoundDevice
foreach ($device in $sound) {
    Write-Host "Ses Cihazi: $($device.Name)"
    Write-Host "  Uretici: $($device.Manufacturer)"
    Write-Host "  Durum: $($device.Status)"
    Write-Host ""
}

# USB Cihazlari
Write-Host "=== USB CIHAZLARI ===" -ForegroundColor Yellow
$usb = Get-CimInstance Win32_USBControllerDevice
$usbDevices = @()
foreach ($device in $usb) {
    $usbInfo = $device.Dependent
    if ($usbInfo) {
        $usbDevices += Get-CimInstance -Query "SELECT * FROM Win32_PnPEntity WHERE DeviceID='$($usbInfo.DeviceID.Replace('\','\\'))'" -ErrorAction SilentlyContinue
    }
}
$usbDevices | Select-Object -Unique Name, Manufacturer, Status | Format-Table -AutoSize

# Pil Bilgileri (Laptop ise)
Write-Host "=== PIL BILGILERI (Laptop icin) ===" -ForegroundColor Yellow
$battery = Get-CimInstance Win32_Battery
if ($battery) {
    foreach ($bat in $battery) {
        Write-Host "Pil Durumu: $($bat.Status)"
        Write-Host "Sarj Durumu: $($bat.BatteryStatus)"
        Write-Host "Kalan Sarj: $($bat.EstimatedChargeRemaining)%"
        Write-Host "Tahmini Calisma Suresi: $($bat.EstimatedRunTime) dakika"
        Write-Host "Tasarim Kapasitesi: $($bat.DesignCapacity)"
        Write-Host "Tam Sarj Kapasitesi: $($bat.FullChargeCapacity)"
        Write-Host ""
    }
} else {
    Write-Host "Pil bulunamadi (Masaustu bilgisayar)" -ForegroundColor Gray
    Write-Host ""
}

# Sistem Performans Bilgileri
Write-Host "=== PERFORMANS BILGILERI ===" -ForegroundColor Yellow
try {
    $cpuCounter = Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction Stop
    Write-Host "CPU Kullanimi: $($cpuCounter.CounterSamples.CookedValue.ToString('N2'))%"
} catch {
    Write-Host "CPU kullanim bilgisi alinamadi" -ForegroundColor Gray
}

try {
    $mem = Get-Counter '\Memory\Available MBytes' -ErrorAction Stop
    $totalMem = $cs.TotalPhysicalMemory / 1MB
    $usedMem = $totalMem - $mem.CounterSamples.CookedValue
    Write-Host "Kullanilan RAM: $([math]::Round($usedMem / 1024, 2)) GB / $([math]::Round($totalMem / 1024, 2)) GB"
    Write-Host "Bos RAM: $([math]::Round($mem.CounterSamples.CookedValue / 1024, 2)) GB"
} catch {
    Write-Host "RAM kullanim bilgisi alinamadi" -ForegroundColor Gray
}
Write-Host ""

# Guc Kaynagi Bilgileri
Write-Host "=== GUC KAYNAGI ===" -ForegroundColor Yellow
$power = Get-CimInstance Win32_PowerSupply -ErrorAction SilentlyContinue
if ($power) {
    foreach ($ps in $power) {
        Write-Host "Durum: $($ps.Status)"
        Write-Host "Aciklama: $($ps.Description)"
        Write-Host ""
    }
} else {
    Write-Host "Guc kaynagi bilgisi alinamadi" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "          RAPOR TAMAMLANDI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Bu raporu bir dosyaya kaydetmek icin:" -ForegroundColor Green
Write-Host ".\get_hardware_info_fixed.ps1 | Out-File -FilePath hardware_report.txt -Encoding UTF8" -ForegroundColor Green
