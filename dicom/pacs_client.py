"""
PACS (Picture Archiving and Communication System) client module.
Provides DICOM networking capabilities: C-ECHO, C-FIND, C-MOVE, C-GET
for querying and retrieving medical images from PACS servers.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

try:
    from pynetdicom import AE, evt, StoragePresentationContexts
    from pynetdicom.sop_class import (
        PatientRootQueryRetrieveInformationModelFind,
        PatientRootQueryRetrieveInformationModelMove,
        PatientRootQueryRetrieveInformationModelGet,
        StudyRootQueryRetrieveInformationModelFind,
        StudyRootQueryRetrieveInformationModelMove,
        Verification,
    )
    from pydicom.dataset import Dataset
    from pydicom.uid import ExplicitVRLittleEndian
    HAS_PYNETDICOM = True
except ImportError:
    HAS_PYNETDICOM = False

logger = logging.getLogger(__name__)


@dataclass
class PACSConfig:
    """Configuration for PACS server connection."""
    host: str = "127.0.0.1"
    port: int = 11112
    ae_title: str = "MEDVLM_SCU"
    peer_ae_title: str = "PACS_SCP"
    local_port: int = 11113
    retrieve_directory: str = "./dicom_retrieved"
    timeout: int = 30


@dataclass
class PACSQueryResult:
    """Represents a single query result from PACS."""
    patient_name: str = ""
    patient_id: str = ""
    study_date: str = ""
    study_description: str = ""
    modality: str = ""
    accession_number: str = ""
    study_instance_uid: str = ""
    series_instance_uid: str = ""
    number_of_images: int = 0


class PACSClient:
    """
    PACS client for querying and retrieving medical images.
    Supports C-ECHO, C-FIND, C-MOVE, and C-GET operations.
    """

    def __init__(self, config: Optional[PACSConfig] = None):
        if not HAS_PYNETDICOM:
            raise ImportError(
                "pynetdicom is required for PACS connectivity. "
                "Install it with: pip install pynetdicom"
            )

        self.config = config or PACSConfig()
        self.ae = AE(ae_title=self.config.ae_title)
        self._retrieved_files = []

        os.makedirs(self.config.retrieve_directory, exist_ok=True)

    def verify_connection(self):
        """
        Send C-ECHO to verify PACS server connectivity.

        Returns:
            tuple: (success: bool, message: str)
        """
        self.ae.add_requested_context(Verification)

        try:
            assoc = self.ae.associate(
                self.config.host,
                self.config.port,
                ae_title=self.config.peer_ae_title,
            )

            if assoc.is_established:
                status = assoc.send_c_echo()
                assoc.release()

                if status and status.Status == 0x0000:
                    return True, "PACS 連線成功 (Connection successful)"
                return False, f"C-ECHO 失敗, 狀態碼: {status.Status if status else 'None'}"

            return False, "無法建立 DICOM 關聯 (Association failed)"

        except Exception as e:
            return False, f"連線錯誤 (Connection error): {str(e)}"

    def query_studies(self, patient_name="", patient_id="",
                      study_date="", modality="", accession_number=""):
        """
        Query PACS for studies using C-FIND.

        Args:
            patient_name: Patient name (supports wildcards *)
            patient_id: Patient ID
            study_date: Study date (YYYYMMDD, supports ranges e.g. 20230101-20231231)
            modality: Modality code (CR, CT, MR, US, etc.)
            accession_number: Accession number

        Returns:
            list of PACSQueryResult
        """
        self.ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)

        # Build query dataset
        ds = Dataset()
        ds.QueryRetrieveLevel = 'STUDY'
        ds.PatientName = patient_name or '*'
        ds.PatientID = patient_id or ''
        ds.StudyDate = study_date or ''
        ds.ModalitiesInStudy = modality or ''
        ds.StudyDescription = ''
        ds.AccessionNumber = accession_number or ''
        ds.StudyInstanceUID = ''
        ds.NumberOfStudyRelatedSeries = ''
        ds.NumberOfStudyRelatedInstances = ''

        results = []

        try:
            assoc = self.ae.associate(
                self.config.host,
                self.config.port,
                ae_title=self.config.peer_ae_title,
            )

            if assoc.is_established:
                responses = assoc.send_c_find(
                    ds, StudyRootQueryRetrieveInformationModelFind
                )

                for status, identifier in responses:
                    if status and status.Status in (0xFF00, 0xFF01) and identifier:
                        result = PACSQueryResult(
                            patient_name=str(getattr(identifier, 'PatientName', '')),
                            patient_id=str(getattr(identifier, 'PatientID', '')),
                            study_date=str(getattr(identifier, 'StudyDate', '')),
                            study_description=str(getattr(identifier, 'StudyDescription', '')),
                            modality=str(getattr(identifier, 'ModalitiesInStudy', '')),
                            accession_number=str(getattr(identifier, 'AccessionNumber', '')),
                            study_instance_uid=str(getattr(identifier, 'StudyInstanceUID', '')),
                            number_of_images=int(getattr(identifier, 'NumberOfStudyRelatedInstances', 0) or 0),
                        )
                        results.append(result)

                assoc.release()
            else:
                logger.error("Failed to establish association for C-FIND")

        except Exception as e:
            logger.error(f"C-FIND error: {e}")

        return results

    def query_series(self, study_instance_uid):
        """
        Query series within a study.

        Args:
            study_instance_uid: The Study Instance UID to query

        Returns:
            list of PACSQueryResult with series-level info
        """
        self.ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)

        ds = Dataset()
        ds.QueryRetrieveLevel = 'SERIES'
        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = ''
        ds.Modality = ''
        ds.SeriesDescription = ''
        ds.SeriesNumber = ''
        ds.NumberOfSeriesRelatedInstances = ''
        ds.BodyPartExamined = ''

        results = []

        try:
            assoc = self.ae.associate(
                self.config.host,
                self.config.port,
                ae_title=self.config.peer_ae_title,
            )

            if assoc.is_established:
                responses = assoc.send_c_find(
                    ds, StudyRootQueryRetrieveInformationModelFind
                )

                for status, identifier in responses:
                    if status and status.Status in (0xFF00, 0xFF01) and identifier:
                        result = PACSQueryResult(
                            modality=str(getattr(identifier, 'Modality', '')),
                            study_description=str(getattr(identifier, 'SeriesDescription', '')),
                            study_instance_uid=study_instance_uid,
                            series_instance_uid=str(getattr(identifier, 'SeriesInstanceUID', '')),
                            number_of_images=int(getattr(identifier, 'NumberOfSeriesRelatedInstances', 0) or 0),
                        )
                        results.append(result)

                assoc.release()

        except Exception as e:
            logger.error(f"C-FIND series error: {e}")

        return results

    def retrieve_study(self, study_instance_uid):
        """
        Retrieve all images for a study using C-MOVE.

        Args:
            study_instance_uid: The Study Instance UID to retrieve

        Returns:
            tuple: (success: bool, file_paths: list, message: str)
        """
        self.ae.add_requested_context(StudyRootQueryRetrieveInformationModelMove)

        # Also set up as SCP to receive the moved images
        self._retrieved_files = []
        storage_dir = os.path.join(
            self.config.retrieve_directory, study_instance_uid[:16]
        )
        os.makedirs(storage_dir, exist_ok=True)

        def handle_store(event):
            """Handle incoming C-STORE requests from PACS."""
            ds = event.dataset
            ds.file_meta = event.file_meta
            sop_uid = ds.SOPInstanceUID
            filepath = os.path.join(storage_dir, f"{sop_uid}.dcm")
            ds.save_as(filepath)
            self._retrieved_files.append(filepath)
            return 0x0000

        handlers = [(evt.EVT_C_STORE, handle_store)]

        # Start SCP on local port for receiving
        scp = AE(ae_title=self.config.ae_title)
        for cx in StoragePresentationContexts:
            scp.add_supported_context(cx.abstract_syntax)

        scp_server = scp.start_server(
            ('', self.config.local_port),
            block=False,
            evt_handlers=handlers,
        )

        try:
            ds = Dataset()
            ds.QueryRetrieveLevel = 'STUDY'
            ds.StudyInstanceUID = study_instance_uid

            assoc = self.ae.associate(
                self.config.host,
                self.config.port,
                ae_title=self.config.peer_ae_title,
            )

            if assoc.is_established:
                responses = assoc.send_c_move(
                    ds,
                    self.config.ae_title,
                    StudyRootQueryRetrieveInformationModelMove,
                )

                for status, identifier in responses:
                    if status and status.Status == 0x0000:
                        break

                assoc.release()

                return (
                    True,
                    self._retrieved_files,
                    f"成功接收 {len(self._retrieved_files)} 個影像檔案"
                )
            else:
                return False, [], "無法建立 DICOM 關聯"

        except Exception as e:
            return False, [], f"C-MOVE 錯誤: {str(e)}"

        finally:
            scp_server.shutdown()

    def get_connection_status_text(self, success, message):
        """Format connection status for GUI display."""
        status_icon = "✅" if success else "❌"
        return (
            f"{status_icon} {message}\n"
            f"  伺服器 Host: {self.config.host}:{self.config.port}\n"
            f"  AE Title:    {self.config.peer_ae_title}\n"
            f"  本機 Local:  {self.config.ae_title}"
        )

    def format_query_results_table(self, results):
        """Format query results as a displayable table string."""
        if not results:
            return "查無結果 (No results found)"

        lines = []
        lines.append(f"共找到 {len(results)} 筆研究 (Found {len(results)} studies)\n")
        lines.append("-" * 80)
        lines.append(
            f"{'#':<4} {'病患姓名':<16} {'檢查日期':<12} "
            f"{'模態':<6} {'描述':<20} {'影像數':<8}"
        )
        lines.append("-" * 80)

        for i, r in enumerate(results, 1):
            lines.append(
                f"{i:<4} {r.patient_name[:14]:<16} {r.study_date:<12} "
                f"{r.modality:<6} {r.study_description[:18]:<20} {r.number_of_images:<8}"
            )

        lines.append("-" * 80)
        return "\n".join(lines)
