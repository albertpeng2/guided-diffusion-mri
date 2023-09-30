import testing.fastmri_class as fastmri

train_dataset = fastmri.RealMeasurement(
                idx_list=range(564, 1100),
                acceleration_rate=2,
                is_return_y_smps_hat=True,
                mask_pattern='uniformly_cartesian',
                smps_hat_method='eps',
            )

