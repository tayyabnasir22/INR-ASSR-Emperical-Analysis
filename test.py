from Validators.EDSR_LIIF_SimplePairedValidator import EDSR_LIIF_SimplePairedValidator
from Validators.EDSR_LIIF_SimpleValidator import EDSR_LIIF_SimpleValidator
from Validators.EDSR_LIIF_PatchedValidator import EDSR_LIIF_PatchedValidator
from Validators.EDSR_LIIF_OverlapPatchedValidator import EDSR_LIIF_OverlapPatchedValidator

def main():
    # EDSR_LIIF_SimplePairedValidator().TestModel()

    # EDSR_LIIF_SimpleValidator().TestModel()

    # EDSR_LIIF_PatchedValidator().TestModel()

    EDSR_LIIF_OverlapPatchedValidator().TestModel()


if __name__ == '__main__':
    main()