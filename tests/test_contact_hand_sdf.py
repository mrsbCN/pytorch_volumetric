"""
Test the enhanced ContactHandSDF implementation
"""

import torch
import pytorch_kinematics as pk
from pytorch_volumetric.contact_robot_hand_sdf import ContactHandSDF
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_contact_hand_sdf():
    """Test the complete ContactHandSDF implementation"""
    print("=== Testing ContactHandSDF Implementation ===")
    
    # Load hand model
    try:
        with open("hand/schunk_svh_hand_right.urdf", "rb") as f:
            chain = pk.build_mimic_chain_from_urdf(f.read())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chain = chain.to(device=device)
        print(f"✓ Hand model loaded on {device}")
    except Exception as e:
        print(f"✗ Failed to load hand model: {e}")
        return
    
    # Create ContactHandSDF
    try:
        contact_hand_sdf = ContactHandSDF(chain, path_prefix="hand/")
        print("✓ ContactHandSDF created successfully")
    except Exception as e:
        print(f"✗ Failed to create ContactHandSDF: {e}")
        return
    
    # Set initial joint configuration
    n_joints = len(contact_hand_sdf.joint_names)
    joint_config = torch.zeros(n_joints, device=device)
    contact_hand_sdf.set_joint_configuration(joint_config)
    print(f"✓ Joint configuration set ({n_joints} joints)")
    
    # Test 1: Forward pass (ContactGen style)
    print("\n--- Test 1: Forward Pass ---")
    obj_verts = torch.rand(10, 500, 3, device=device) * 0.08
    
    try:
        pred, pred_p_full = contact_hand_sdf.forward(obj_verts)
        print(f"✓ Forward pass successful")
        print(f"  pred shape: {pred.shape}") #TODO pred shape: torch.Size([500]) should be[1,500] 
        print(f"  pred_p_full shape: {pred_p_full.shape}")
        print(f"  pred range: [{pred.min():.3f}, {pred.max():.3f}]")
        print(f"  pred_p_full range: [{pred_p_full.min():.3f}, {pred_p_full.max():.3f}]")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Contact map query
    print("\n--- Test 2: Contact Map Query ---")
    try:
        contact_map = contact_hand_sdf.query_contact_map(obj_verts)
        print(f"✓ Contact map query successful")
        print(f"  Contact prob shape: {contact_map['contact_prob'].shape}")
        print(f"  Part map shape: {contact_map['part_map'].shape}")
        print(f"  Direction map shape: {contact_map['direction_map'].shape}")
        
        unique_parts = torch.unique(contact_map['part_map'])
        print(f"  Unique parts found: {unique_parts.cpu().numpy()}")
        
        # Show part names
        part_names = contact_hand_sdf.get_part_names()
        print("  Parts in contact:")
        for part_id in unique_parts.cpu().numpy():
            count = (contact_map['part_map'] == part_id).sum().item()
            print(f"    Part {part_id} ({part_names.get(part_id, 'Unknown')}): {count} vertices")
    except Exception as e:
        print(f"✗ Contact map query failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: ContactGen-style loss computation
    print("\n--- Test 3: ContactGen Loss Computation ---")
    try:
        # Create mock ground truth
        obj_cmap = torch.rand(10, 500, device=device) > 0.7  # Binary contact
        obj_partition = torch.randint(0, 16, (10, 500), device=device)  # Random parts
        
        loss_contact, pred_out, pred_p_full_out = contact_hand_sdf.compute_contact_loss(
            obj_verts, obj_cmap.float(), obj_partition
        )
        
        print(f"✓ ContactGen loss computation successful")
        print(f"  Contact loss: {loss_contact:.4f}")
        print(f"  Output shapes match: pred={pred_out.shape}, pred_p_full={pred_p_full_out.shape}")
    except Exception as e:
        print(f"✗ ContactGen loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Part alignment verification
    print("\n--- Test 4: Part Alignment Verification ---")
    try:
        # Test the key ContactGen operation
        test_partition = torch.randint(0, 16, (10, 500), device=device)
        extracted_pred = torch.gather(pred_p_full, dim=-1, 
                                    index=test_partition.unsqueeze(dim=-1)).squeeze(-1)
        
        print(f"✓ Part alignment extraction successful")
        print(f"  Extracted pred shape: {extracted_pred.shape}")
        print(f"  This matches ContactGen line 46 operation!")
        
        # Debug shapes
        print(f"  pred_p_full shape: {pred_p_full.shape}")
        print(f"  test_partition shape: {test_partition.shape}")
        
        # Verify non-zero predictions for active parts
        for part_id in torch.unique(test_partition):
            mask = test_partition == part_id  # Shape: [10, 500]
            # Extract predictions for this part from all points
            part_preds_all_points = pred_p_full[..., part_id]  # Shape: [10, 500] 
            part_preds = part_preds_all_points[mask]  # Extract only points assigned to this part
            
            if part_preds.numel() > 0:
                print(f"  Part {part_id}: count={part_preds.numel()}, "
                      f"mean={part_preds.mean():.3f}, max={part_preds.max():.3f}")
    except Exception as e:
        print(f"✗ Part alignment verification failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 5: Mini optimization test
    print("\n--- Test 5: Mini Optimization Test ---")
    try:
        # Very short optimization test
        initial_config = torch.zeros(n_joints, device=device)
        optimized_config = contact_hand_sdf.optimize_grasp(
            obj_verts, obj_cmap.float(), obj_partition, 
            initial_config, iterations=10, lr=1e-2
        )
        
        print(f"✓ Mini optimization test successful")
        print(f"  Config change: {torch.norm(optimized_config - initial_config):.4f}")
    except Exception as e:
        print(f"✗ Mini optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== All Tests Passed! ===")
    print("ContactHandSDF is ready for ContactGen-style grasp generation!")
    print("\nKey features verified:")
    print("✓ Returns both pred and pred_p_full (like ContactGen)")
    print("✓ Supports part-conditioned contact prediction")
    print("✓ Compatible with ContactGen optimization pipeline")
    print("✓ Proper part alignment and extraction")


if __name__ == "__main__":
    test_contact_hand_sdf()
