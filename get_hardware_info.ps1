# Windows Donanım Bilgileri Raporu
# Detaylı sistem donanım bilgilerini toplar ve gösterir

# Encoding ayarları - Türkçe karakterlerin düzgün görünmesi için
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   WINDOWS DONANIM BİLGİLERİ RAPORU" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# İşletim Sistemi Bilgileri
Write-Host "=== İŞLETİM SİSTEMİ ===" -ForegroundColor Yellow
$os = Get-CimInstance Win32_OperatingSystem
Write-Host "İşletim Sistemi: $($os.Caption)"
Write-Host "Versiyon: $($os.Version)"
Write-Host "Mimari: $($os.OSArchitecture)"
Write-Host "Kurulum Tarihi: $($os.InstallDate)"
Write-Host "Son Açılış: $($os.LastBootUpTime)"
Write-Host "Sistem Dizini: $($os.SystemDirectory)"
Write-Host ""

# Bilgisayar Bilgileri
Write-Host "=== BİLGİSAYAR BİLGİLERİ ===" -ForegroundColor Yellow
$cs = Get-CimInstance Win32_ComputerSystem
Write-Host "Bilgisayar Adı: $($cs.Name)"
Write-Host "Üretici: $($cs.Manufacturer)"
Write-Host "Model: $($cs.Model)"
Write-Host "Sistem Tipi: $($cs.SystemType)"
Write-Host "Domain: $($cs.Domain)"
Write-Host ""

# İşlemci Bilgileri
Write-Host "=== İŞLEMCİ (CPU) ===" -ForegroundColor Yellow
$cpu = Get-CimInstance Win32_Processor
foreach ($proc in $cpu) {
    Write-Host "İşlemci Adı: $($proc.Name)"
    Write-Host "Üretici: $($proc.Manufacturer)"
    Write-Host "Çekirdek Sayısı: $($proc.NumberOfCores)"
    Write-Host "Mantıksal İşlemci: $($proc.NumberOfLogicalProcessors)"
    Write-Host "Maksimum Hız: $($proc.MaxClockSpeed) MHz"
    Write-Host "Mevcut Hız: $($proc.CurrentClockSpeed) MHz"
    Write-Host "L2 Önbellek: $($proc.L2CacheSize) KB"
    Write-Host "L3 Önbellek: $($proc.L3CacheSize) KB"
    Write-Host "Soket Tanımı: $($proc.SocketDesignation)"
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
    Write-Host "  Hız: $($mem.Speed) MHz"
    Write-Host "  Üretici: $($mem.Manufacturer)"
    Write-Host "  Part Number: $($mem.PartNumber)"
    Write-Host "  Seri No: $($mem.SerialNumber)"
    Write-Host "  Form Faktör: $($mem.FormFactor)"
    Write-Host "  Bellek Tipi: $($mem.MemoryType)"
}
Write-Host ""

# Anakart Bilgileri
Write-Host "=== ANAKART (MOTHERBOARD) ===" -ForegroundColor Yellow
$mb = Get-CimInstance Win32_BaseBoard
Write-Host "Üretici: $($mb.Manufacturer)"
Write-Host "Ürün: $($mb.Product)"
Write-Host "Seri No: $($mb.SerialNumber)"
Write-Host "Versiyon: $($mb.Version)"
Write-Host ""

# BIOS Bilgileri
Write-Host "=== BIOS ===" -ForegroundColor Yellow
$bios = Get-CimInstance Win32_BIOS
Write-Host "Üretici: $($bios.Manufacturer)"
Write-Host "Versiyon: $($bios.Version)"
Write-Host "Seri No: $($bios.SerialNumber)"
Write-Host "Tarih: $($bios.ReleaseDate)"
Write-Host ""

# Disk Bilgileri
Write-Host "=== DİSKLER (STORAGE) ===" -ForegroundColor Yellow
$disks = Get-CimInstance Win32_DiskDrive
foreach ($disk in $disks) {
    Write-Host ""
    Write-Host "Disk: $($disk.Caption)"
    Write-Host "  Model: $($disk.Model)"
    Write-Host "  Boyut: $([math]::Round($disk.Size / 1GB, 2)) GB"
    Write-Host "  Arayüz: $($disk.InterfaceType)"
    Write-Host "  Bölüm Sayısı: $($disk.Partitions)"
    Write-Host "  Seri No: $($disk.SerialNumber)"
    Write-Host "  Durum: $($disk.Status)"
}
Write-Host ""

# Bölüm (Partition) Bilgileri
Write-Host "=== BÖLÜMLER (PARTITIONS) ===" -ForegroundColor Yellow
$volumes = Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3"
foreach ($vol in $volumes) {
    Write-Host ""
    Write-Host "Sürücü: $($vol.DeviceID)"
    Write-Host "  Dosya Sistemi: $($vol.FileSystem)"
    Write-Host "  Toplam Boyut: $([math]::Round($vol.Size / 1GB, 2)) GB"
    Write-Host "  Boş Alan: $([math]::Round($vol.FreeSpace / 1GB, 2)) GB"
    Write-Host "  Kullanılan: $([math]::Round(($vol.Size - $vol.FreeSpace) / 1GB, 2)) GB"
    Write-Host "  Kullanım Oranı: $([math]::Round((($vol.Size - $vol.FreeSpace) / $vol.Size) * 100, 2))%"
    Write-Host "  Etiket: $($vol.VolumeName)"
}
Write-Host ""

# Ekran Kartı (GPU) Bilgileri
Write-Host "=== EKRAN KARTI (GPU) ===" -ForegroundColor Yellow
$gpu = Get-CimInstance Win32_VideoController
foreach ($card in $gpu) {
    Write-Host ""
    Write-Host "Ekran Kartı: $($card.Name)"
    Write-Host "  Üretici: $($card.AdapterCompatibility)"
    Write-Host "  Video Belleği: $([math]::Round($card.AdapterRAM / 1GB, 2)) GB"
    Write-Host "  Sürücü Versiyonu: $($card.DriverVersion)"
    Write-Host "  Mevcut Çözünürlük: $($card.CurrentHorizontalResolution) x $($card.CurrentVerticalResolution)"
    Write-Host "  Yenileme Hızı: $($card.CurrentRefreshRate) Hz"
    Write-Host "  Durum: $($card.Status)"
}
Write-Host ""

# Monitör Bilgileri
Write-Host "=== MONİTÖR ===" -ForegroundColor Yellow
$monitors = Get-CimInstance WmiMonitorID -Namespace root\wmi -ErrorAction SilentlyContinue
if ($monitors) {
    foreach ($mon in $monitors) {
        $manufacturer = [System.Text.Encoding]::ASCII.GetString($mon.ManufacturerName -ne 0)
        $name = [System.Text.Encoding]::ASCII.GetString($mon.UserFriendlyName -ne 0)
        $serial = [System.Text.Encoding]::ASCII.GetString($mon.SerialNumberID -ne 0)
        Write-Host "Üretici: $manufacturer"
        Write-Host "Model: $name"
        Write-Host "Seri No: $serial"
        Write-Host ""
    }
} else {
    Write-Host "Monitör bilgisi alınamadı" -ForegroundColor Red
    Write-Host ""
}

# Ağ Adaptörleri
Write-Host "=== AĞ ADAPTÖRLERI ===" -ForegroundColor Yellow
$adapters = Get-CimInstance Win32_NetworkAdapter | Where-Object {$_.PhysicalAdapter -eq $true}
foreach ($adapter in $adapters) {
    Write-Host ""
    Write-Host "Adaptör: $($adapter.Name)"
    Write-Host "  MAC Adresi: $($adapter.MACAddress)"
    Write-Host "  Hız: $($adapter.Speed)"
    Write-Host "  Üretici: $($adapter.Manufacturer)"
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

# Ses Kartı
Write-Host "=== SES KARTI ===" -ForegroundColor Yellow
$sound = Get-CimInstance Win32_SoundDevice
foreach ($device in $sound) {
    Write-Host "Ses Cihazı: $($device.Name)"
    Write-Host "  Üretici: $($device.Manufacturer)"
    Write-Host "  Durum: $($device.Status)"
    Write-Host ""
}

# USB Cihazları
Write-Host "=== USB CİHAZLARI ===" -ForegroundColor Yellow
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
Write-Host "=== PİL BİLGİLERİ (Laptop için) ===" -ForegroundColor Yellow
$battery = Get-CimInstance Win32_Battery
if ($battery) {
    foreach ($bat in $battery) {
        Write-Host "Pil Durumu: $($bat.Status)"
        Write-Host "Şarj Durumu: $($bat.BatteryStatus)"
        Write-Host "Kalan Şarj: $($bat.EstimatedChargeRemaining)%"
        Write-Host "Tahmini Çalışma Süresi: $($bat.EstimatedRunTime) dakika"
        Write-Host "Tasarım Kapasitesi: $($bat.DesignCapacity)"
        Write-Host "Tam Şarj Kapasitesi: $($bat.FullChargeCapacity)"
        Write-Host ""
    }
} else {
    Write-Host "Pil bulunamadı (Masaüstü bilgisayar)" -ForegroundColor Gray
    Write-Host ""
}

# Sistem Performans Bilgileri
Write-Host "=== PERFORMANS BİLGİLERİ ===" -ForegroundColor Yellow
Write-Host "CPU Kullanımı: $((Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue.ToString('N2'))%"
$mem = Get-Counter '\Memory\Available MBytes'
$totalMem = $cs.TotalPhysicalMemory / 1MB
$usedMem = $totalMem - $mem.CounterSamples.CookedValue
Write-Host "Kullanılan RAM: $([math]::Round($usedMem / 1024, 2)) GB / $([math]::Round($totalMem / 1024, 2)) GB"
Write-Host "Boş RAM: $([math]::Round($mem.CounterSamples.CookedValue / 1024, 2)) GB"
Write-Host ""

# Güç Kaynağı Bilgileri
Write-Host "=== GÜÇ KAYNAĞI ===" -ForegroundColor Yellow
$power = Get-CimInstance Win32_PowerSupply -ErrorAction SilentlyContinue
if ($power) {
    foreach ($ps in $power) {
        Write-Host "Durum: $($ps.Status)"
        Write-Host "Açıklama: $($ps.Description)"
        Write-Host ""
    }
} else {
    Write-Host "Güç kaynağı bilgisi alınamadı" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "          RAPOR TAMAMLANDI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Bu raporu bir dosyaya kaydetmek için:" -ForegroundColor Green
Write-Host ".\get_hardware_info.ps1 | Out-File -FilePath hardware_report.txt -Encoding UTF8" -ForegroundColor Green
