"""Pure receipt/acceptance checks. No I/O, process execution or resize entrypoint."""
BACKUP_COMMIT = '2ec4c3dcdbe387127b16c4ca281eb0d1d76d8fa4'
BACKUP_SHA = '36d8679eaad94fdfa7b0cc12a48c728ee4ec73f83de90d339005295852365c5c'
BACKUP_BYTES = 29_581_377_536
SOURCE_PATH = r'D:\AI\Modly-WSL\Ubuntu24.04\ext4.vhdx'
BACKUP_PATH = r'E:\Modly-Cold-Backup-closed-solid-v2\ext4.vhdx'
GIB = 1024**3
TARGET_BYTES = 64*GIB
NATIVE_COMMAND = ('C:\\Windows\\System32\\wsl.exe', '--manage',
                  'Modly-AMD-Ubuntu-24.04', '--resize', '68719476736B')


def require(condition, message):
    if not condition: raise ValueError(message)


def verify_backup(final, worker, independent, jobs, current):
    """Cross-check saved receipts plus separately observed current metadata.

    This is not a fresh full-file hash and cannot establish filesystem health.
    Callers must bind input files/metadata provenance; fixtures never run WSL.
    """
    require(final.get('classification') == 'PASS' and worker.get('classification') == 'PASS', 'Both final and worker PASS required')
    require(final.get('commit') == BACKUP_COMMIT, 'Exact completed backup commit required')
    require(final.get('source') == SOURCE_PATH and final.get('destination') == BACKUP_PATH, 'Fixed backup paths required')
    require(final.get('verification') == worker, 'Final embedded worker differs')
    require(final.get('controller_handles_closed') is True and worker.get('handles_closed') is True, 'Handle cleanup unverified')
    require(worker.get('bytes') == BACKUP_BYTES, 'Backup length differs')
    for key in ('copy_sha256', 'source_sha256', 'destination_sha256'):
        require(worker.get(key) == BACKUP_SHA, 'Fixed authenticated SHA-256 differs: '+key)
    require(independent.get('size') == BACKUP_BYTES and independent.get('sha256') == BACKUP_SHA, 'Independent hash receipt differs')
    require(worker.get('source_identity') == worker.get('source_after') and worker.get('source_identity') is not None, 'Source changed')
    require(worker['source_identity'].get('final_path') == SOURCE_PATH and worker['destination_identity'].get('final_path') == BACKUP_PATH, 'Handle path differs')
    require(len(jobs) == 3, 'All three owned Job receipts required')
    for row in jobs:
        cleanup = row.get('cleanup', {})
        require(row.get('returncode') == 0 and row.get('job_attached') is True, 'Owned process failed')
        require(type(row.get('peak_job_commit_bytes')) is int and row['peak_job_commit_bytes'] <= 512*1024**2, 'Job memory receipt differs')
        require(cleanup.get('owned_tree_exit_verified') is True and cleanup.get('forced_job_cleanup') is False
                and cleanup.get('active_after_cleanup') == 0, 'Natural cleanup unverified')
    for name, identity in (('source', worker['source_identity']), ('backup', worker['destination_identity'])):
        require(identity.get('size') == BACKUP_BYTES and current[name]['identity'] == identity, 'Current file identity differs: '+name)
        meta = worker['source_metadata' if name == 'source' else 'destination_metadata']
        require(current[name]['metadata'] == meta and meta.get('virtual_capacity_bytes') == 32*GIB, 'Current capacity/metadata differs: '+name)
    require(worker['source_metadata'] == worker['destination_metadata'], 'Copied metadata differs')
    return dict(classification='BACKUP_RECEIPTS_AND_METADATA_PASS', sha256=BACKUP_SHA, bytes=BACKUP_BYTES)


def require_native_ram(available):
    require(type(available) is int and available >= 22*GIB, '22 GiB required throughout native invocation for PASS')


def accept_completion(row):
    """Offline acceptance only; missing facts cannot promote partial growth.

    filesystem_bytes must be established by native filesystem block count and
    block size, not equated to df usable bytes or VHDX capacity by assumption.
    """
    require(row.get('native_exit') == 0 and row.get('natural_completion') is True, 'Native completion unverified')
    require(row.get('guard_faults') == [], 'Any latched guard fault prevents PASS')
    for key in ('virtual_bytes', 'filesystem_bytes', 'root_device_bytes'):
        require(type(row.get(key)) is int and row[key] == TARGET_BYTES, 'Exact 64 GiB not established: '+key)
    for key in ('ext4_rw', 'backup_unchanged', 'registration_unchanged', 'inputs_unchanged'):
        require(row.get(key) is True, 'Preservation/mount condition failed: '+key)
    for key, floor in (('available_bytes', 5*GIB), ('D_free_bytes', 50*GIB)):
        require(type(row.get(key)) is int and row[key] >= floor, 'Disk floor failed: '+key)
    return True


if __name__ == '__main__':
    print('OFFLINE_ONLY: no resize executor. Critical-section policy approval and fresh preflight required.')
