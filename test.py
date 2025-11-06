from Validators.EDSR_LIIF_SimplePairedValidator import EDSR_LIIF_SimplePairedValidator
from Validators.EDSR_LIIF_SimpleValidator import EDSR_LIIF_SimpleValidator
from Validators.EDSR_LIIF_PatchedValidator import EDSR_LIIF_PatchedValidator
from Validators.EDSR_LIIF_OverlapPatchedValidator import EDSR_LIIF_OverlapPatchedValidator
from Validators.EDSR_LIIF_PatchedPairedValidator import EDSR_LIIF_PatchedPairedValidator
from Validators.EDSR_LIIF_OverlapPatchedPairedValidator import EDSR_LIIF_OverlapPatchedPairedValidator

def main():
    # EDSR_LIIF_SimplePairedValidator().TestModel()

    # EDSR_LIIF_SimpleValidator().TestModel()

    # EDSR_LIIF_PatchedValidator().TestModel()

    # EDSR_LIIF_OverlapPatchedValidator().TestModel()

    # EDSR_LIIF_PatchedPairedValidator().TestModel()

    EDSR_LIIF_OverlapPatchedPairedValidator().TestModel()


if __name__ == '__main__':
    main()